"""Responsibilities: Implement independent asynchronous multimodal model adapter for resume cleanup
Agent.
Implementation: Read dedicated configuration, pass single-page PNG and numbered source lines in Chat
Completions JSON format, strictly validate structure. Explicit free_pool mode restricts rotation to
approved vision-capable models; text-only models cannot enter the PDF cleanup route.
Related Modules: agents.resume_cleanup defines the port; resume_api manages adapter lifecycle;
app.providers.free_pool selects approved vision candidates using the shared private manifest/state.
Declaration Index:
- ResumeVision: OpenAI-compatible vision client with optional explicit free-quota vision routing.
- ResumeVision.__init__: Validate dedicated vision configuration, create async client.
- ResumeVision.review: Pass single-page image and text, verify final model state and structure.
- ResumeVision.close: Asynchronously release client connection.
Variable Index:
- logger: Logs only page number, model, duration, and exception type, no image, resume, or key data.
Configuration Notes:
RESUME_VISION_PROVIDER must be dashscope or openai, RESUME_VISION_MODEL required;
Credentials reuse existing API_KEY from selected provider, BASE_URL matches provider’s existing
configuration.
RESUME_VISION_TIMEOUT_SECONDS defaults to 90 seconds; one logical review per page, max_retries=0.
Enabled free_pool shares that budget across explicit quota/access transitions.
RESUME_VISION_ENABLE_THINKING can be explicitly set true/false, only DashScope accepts this optional
parameter.
Unset value retains provider behavior; does not modify existing interview model, temperature, or
timeout configuration.
"""

import base64
import json
import logging
import os
from time import perf_counter

from openai import APITimeoutError, AsyncOpenAI
from pydantic import ValidationError

from agents.model_calls import safe_error_details, validation_issues
from agents.resume_cleanup import PageReview, ResumePage, source_lines
from app.providers.free_pool import complete_async, require_provider

logger = logging.getLogger(__name__)


class ResumeVision:
    """Function: Asynchronous vision port; logic: one logical review per page; constraint: errors
    and cancellations propagate directly.
    """

    def __init__(self):
        """Read process environment configuration, output client instance; missing config raises
        ValueError, no network request initiated.

        model/provider saved selected vision configuration, options contains only explicit thinking
        setting;
        client must be closed by caller in finally block, keys not logged or returned in response.
        """
        self.provider = os.environ.get("RESUME_VISION_PROVIDER", "")
        require_provider(self.provider)
        self.model = os.environ.get("RESUME_VISION_MODEL", "").strip()
        if self.provider not in {"dashscope", "openai"} or not self.model:
            raise ValueError("Set RESUME_VISION_PROVIDER and RESUME_VISION_MODEL.")
        prefix = "DASHSCOPE" if self.provider == "dashscope" else "OPENAI"
        key = os.environ.get(f"{prefix}_API_KEY", "")
        if not key:
            raise ValueError(
                "The selected vision provider has no API key. "
                "Check the backend environment configuration."
            )
        timeout = float(os.environ.get("RESUME_VISION_TIMEOUT_SECONDS", "90"))
        if not 0 < timeout <= 600:
            raise ValueError("RESUME_VISION_TIMEOUT_SECONDS must be between 0 and 600 seconds.")
        thinking = os.environ.get("RESUME_VISION_ENABLE_THINKING", "")
        if thinking not in {"", "true", "false"}:
            raise ValueError("RESUME_VISION_ENABLE_THINKING must be true or false.")
        self.options = {}
        if thinking:
            if self.provider != "dashscope":
                raise ValueError("RESUME_VISION_ENABLE_THINKING is supported only by DashScope.")
            self.options["extra_body"] = {"enable_thinking": thinking == "true"}
        default_url = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if self.provider == "dashscope"
            else "https://api.openai.com/v1"
        )
        self.client = AsyncOpenAI(
            api_key=key,
            base_url=os.environ.get(f"{prefix}_BASE_URL") or default_url,
            timeout=timeout,
            max_retries=0,
        )

    async def review(self, page: ResumePage, prompt: str) -> PageReview:
        """Input image page and system prompt, output schema validation result; no schema repair.
        Explicit free-pool mode advances only approved vision models on terminal model errors.

        Image uses embedded PNG, not uploaded to file service; model must support image and JSON
        schema.
        Reject, truncate, empty output, or non-JSON all fail.
        Source line numbers shared with Agent; only line separators removed, internal whitespace
        preserved.
        Logs contain only diagnostic metadata; provider response body, even on error, is not logged.
        Success records number of modifications; failure records safe error code, HTTP status, and
        sanitized schema path without input values.
        """
        started = perf_counter()
        logger.info("resume_vision start page=%s model=%s", page.number, self.model)
        try:
            response = await complete_async(
                self.client,
                capability="vision",
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                        + "\nJSON schema:\n"
                        + json.dumps(PageReview.model_json_schema(), ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "number": page.number,
                                        "rule_lines": [
                                            {"line": number, "text": line.rstrip("\r\n")}
                                            for number, line in enumerate(
                                                source_lines(page.text), 1
                                            )
                                        ],
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(page.image_png).decode("ascii")
                                },
                            },
                        ],
                    },
                ],
                **self.options,
            )
            if not response.choices:
                raise ValueError("resume_vision_empty_choices")
            choice = response.choices[0]
            if choice.finish_reason != "stop" or choice.message.refusal:
                raise ValueError("resume_vision_incomplete_or_refused")
            result = PageReview.model_validate_json(choice.message.content or "")
            logger.info(
                "resume_vision validated page=%s response_model=%s corrections=%d uncertainties=%d "
                "input_tokens=%s output_tokens=%s",
                page.number,
                getattr(response, "model", self.model),
                len(result.corrections),
                len(result.uncertainties),
                getattr(getattr(response, "usage", None), "prompt_tokens", None),
                getattr(getattr(response, "usage", None), "completion_tokens", None),
            )
            return result
        except Exception as exc:
            details = safe_error_details(exc)
            logger.warning(
                "resume_vision failed page=%s error=%s code=%s status=%s schema_issues=%s",
                page.number,
                type(exc).__name__,
                "timeout" if isinstance(exc, APITimeoutError) else details["category"],
                details["status_code"],
                validation_issues(exc, PageReview) if isinstance(exc, ValidationError) else [],
            )
            raise
        finally:
            logger.info(
                "resume_vision end page=%s duration_ms=%d",
                page.number,
                (perf_counter() - started) * 1000,
            )

    async def close(self):
        """Close async HTTP connection; no file deletion, no model configuration modification."""
        await self.client.close()
