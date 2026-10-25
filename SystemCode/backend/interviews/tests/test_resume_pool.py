"""Responsibilities: Verify resume vision routing against the existing shared pool contract.
Implementation: Use real OpenAI SDK parsing with HTTPX MockTransport, temporary approval/state
files, and the production ResumeVision adapter. No external requests or billing checks occur.
Related Modules: interviews.resume_vision and app.providers.free_pool implement these boundaries.
Declaration Index:
- ResumePoolTests: Exercise image routing, persistent state, and failure/cancellation semantics.
- ResumePoolTests.setUp: Create isolated credentials, approval, and process environment.
- ResumePoolTests.sdk: Build an owned asynchronous SDK with scripted offline responses.
- ResumePoolTests.sdk.handle: Record requests and consume one scripted response per dispatch.
- ResumePoolTests.review: Invoke the real resume adapter with an offline SDK and close it.
- ResumePoolTests.test_vision_rotation_and_restart: Preserve payloads and durable blocks.
- ResumePoolTests.test_nonterminal_failures_do_not_rotate: Propagate unrelated HTTP failures.
- ResumePoolTests.test_exhaustion_has_no_unapproved_escape: Reject empty eligible pools.
- ResumePoolTests.test_invalid_output_does_not_rotate: Reject malformed/truncated output once.
- ResumePoolTests.test_cancel_and_timeout_do_not_rotate: Preserve cancellation and timeout state.
- ResumePoolTests.test_disabled_mode_preserves_fixed_model: Retain the original explicit path.
- ResumePoolTests.test_invalid_approval_fails_before_dispatch: Enforce key/manifest/storage binding.
- ResumePoolTests.test_shared_deadline_limits_rotation: Bound all attempts by one request budget.
- ResumePoolTests.test_late_success_preserves_terminal_state: Protect shared state during races.
- ResumePoolTests.test_provider_and_request_guards: Reject unsupported providers/options.
Variable Index:
- KEY: Synthetic credential used only in offline SDK tests.
- REVIEW: Valid minimal PageReview JSON returned by the offline transport.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from django.test import SimpleTestCase
from openai import APIStatusError, AsyncOpenAI

from agents.resume_cleanup import ResumePage
from app.providers.free_pool import BASE_URL, FreePool, PoolError, complete_async, identity
from interviews.resume_vision import ResumeVision

KEY = "offline-resume-pool-key"
REVIEW = '{"number":1,"corrections":[],"uncertainties":[]}'


class ResumePoolTests(SimpleTestCase):
    """Functionality: Test routing without contacting a provider or using a Django database.
    Logic: Each case owns its manifest and SQLite file; real SDK errors cross the adapter boundary.
    Constraints: Passing cases establish request/state contracts, not actual model availability.
    """

    def setUp(self):
        """Functionality: Establish independent explicit pool approval for each test.
        Inputs: unittest lifecycle. Outputs: temporary paths and restored-on-cleanup environment.
        Logic: Approve two vision models and one text-only model under a synthetic key identity.
        Constraints: Environment is cleared; no local .env or production state can affect results.
        """
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "identity": identity(KEY, BASE_URL),
                    "models": [
                        {"id": "text-only", "capabilities": ["text"], "free_only": True},
                        {"id": "vision-first", "capabilities": ["vision"], "free_only": True},
                        {"id": "vision-next", "capabilities": ["vision"], "free_only": True},
                    ],
                }
            ),
            encoding="utf-8",
        )
        environment = patch.dict(
            "os.environ",
            {
                "FREE_MODEL_POOL_ENABLED": "true",
                "FREE_MODEL_POOL_MANIFEST": str(self.manifest),
                "FREE_MODEL_POOL_STATE": str(self.root / "state.sqlite3"),
                "RESUME_VISION_PROVIDER": "dashscope",
                "RESUME_VISION_MODEL": "vision-first",
                "RESUME_VISION_ENABLE_THINKING": "false",
                "DASHSCOPE_API_KEY": KEY,
                "DASHSCOPE_BASE_URL": BASE_URL,
            },
            clear=True,
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.pool = FreePool(KEY, BASE_URL)

    def sdk(self, script, *, content=REVIEW, finish="stop"):
        """Functionality: Construct an offline client with real SDK protocol validation.
        Inputs: mutable status/code script, success content and finish reason.
        Outputs: owned client and captured request list. Logic: one script item per HTTP request.
        Constraints: Exhausted scripts fail immediately; SDK retries are zero.
        """
        requests = []

        def handle(request):
            """Functionality: Return one scripted provider result and capture its request.
            Inputs: HTTPX request. Outputs: HTTP response. Logic: decode body and consume script.
            Constraints: No network; error messages contain a sentinel to test log redaction.
            """
            body = json.loads(request.content)
            requests.append(body)
            status, code = script.pop(0)
            if status != 200:
                return httpx.Response(status, json={"error": {"code": code, "message": "SECRET"}})
            return httpx.Response(
                200,
                json={
                    "id": "offline",
                    "created": 0,
                    "object": "chat.completion",
                    "model": body["model"],
                    "usage": {"total_tokens": 5},
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish,
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                },
            )

        return AsyncOpenAI(
            api_key=KEY,
            base_url=BASE_URL,
            timeout=10,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ), requests

    async def review(self, client):
        """Functionality: Run the real image adapter against the supplied offline transport.
        Inputs: owned SDK. Outputs: validated PageReview or original exception.
        Logic: Substitute only SDK construction; preserve routing, schema and image encoding.
        Constraints: Client closes even on failure; synthetic bytes are never sent externally.
        """
        with patch("interviews.resume_vision.AsyncOpenAI", return_value=client):
            adapter = ResumeVision()
        try:
            return await adapter.review(
                ResumePage(1, "raw", "line one\nline two", image_png=b"png"), "JSON"
            )
        finally:
            await adapter.close()

    async def test_vision_rotation_and_restart(self):
        """Functionality: Verify model-not-found rotation, exact payload retention and restart skip.
        Inputs: isolated pool and offline 404/success responses. Outputs: requests/state assertions.
        Logic: Reject first model, persist unavailable, then reinstantiate the adapter.
        Constraints: Neither text-only candidates nor secrets enter routing/log output.
        """
        client, requests = self.sdk([(404, "model_not_found"), (200, None)])
        with self.assertLogs(level="INFO") as logs:
            self.assertEqual((await self.review(client)).number, 1)
        self.assertEqual([r["model"] for r in requests], ["vision-first", "vision-next"])
        self.assertEqual(
            {k: v for k, v in requests[0].items() if k != "model"},
            {k: v for k, v in requests[1].items() if k != "model"},
        )
        self.assertEqual(requests[1]["messages"][1]["content"][1]["type"], "image_url")
        self.assertEqual(requests[1]["response_format"], {"type": "json_object"})
        self.assertFalse(requests[1]["enable_thinking"])
        self.assertNotIn("SECRET", str(logs.output))
        self.assertNotIn(KEY, str(logs.output))
        client, requests = self.sdk([(200, None)])
        await self.review(client)
        self.assertEqual([r["model"] for r in requests], ["vision-next"])
        self.assertEqual(self.pool.states()["vision-first"]["status"], "unavailable")

    async def test_nonterminal_failures_do_not_rotate(self):
        """Functionality: Preserve HTTP errors unrelated to model-specific terminal state.
        Inputs: offline auth/rate-limit/arrears/unknown-permission/server/parameter cases.
        Outputs: one request per case and no persisted blocks. Logic: propagate SDK exceptions.
        Constraints: Generic 403 and rate limits must never be treated as exhausted quota.
        """
        for status, code in [
            (401, "invalid_api_key"),
            (429, "Throttling"),
            (400, "Arrearage"),
            (403, "AccessDenied"),
            (500, "InternalError"),
            (400, "InvalidParameter"),
        ]:
            with self.subTest(status=status, code=code):
                client, requests = self.sdk([(status, code)])
                with self.assertRaises(APIStatusError):
                    await self.review(client)
                self.assertEqual(len(requests), 1)
                self.assertEqual(self.pool.states(), {})

    async def test_exhaustion_has_no_unapproved_escape(self):
        """Functionality: Reject exhaustion without calling unapproved or text-only models.
        Inputs: offline terminal quota/model failures. Outputs: PoolError and durable block checks.
        Logic: Exhaust both candidates, then repeat with an unapproved preferred model.
        Constraints: Second invocation sends zero requests and cannot return partial success.
        """
        client, requests = self.sdk([(403, "FreeTierOnly"), (404, "model_not_found")])
        with self.assertRaises(PoolError):
            await self.review(client)
        self.assertEqual(len(requests), 2)
        client, requests = self.sdk([])
        with patch.dict("os.environ", {"RESUME_VISION_MODEL": "unapproved"}):
            with self.assertRaises(PoolError):
                await self.review(client)
        self.assertEqual(requests, [])

    async def test_invalid_output_does_not_rotate(self):
        """Functionality: Keep schema and completion failures strict after pool selection.
        Inputs: malformed JSON and truncated offline results. Outputs: failure with one dispatch.
        Logic: The pool returns the response; ResumeVision rejects it without another model call.
        Constraints: This test does not assess semantic correctness of model corrections.
        """
        for content, finish in [("not json", "stop"), (REVIEW, "length")]:
            client, requests = self.sdk([(200, None)], content=content, finish=finish)
            with self.assertRaises(ValueError):
                await self.review(client)
            self.assertEqual(len(requests), 1)

    async def test_cancel_and_timeout_do_not_rotate(self):
        """Functionality: Propagate interrupted network work without poisoning shared state.
        Inputs: explicit cancellation and timeout exceptions at the awaited SDK boundary.
        Outputs: unchanged exception type and empty state. Logic: mock one awaited call per case.
        Constraints: No sleep, network, automatic retry or further model selection.
        """
        for failure in [asyncio.CancelledError(), TimeoutError()]:
            client = SimpleNamespace(
                api_key=KEY,
                base_url=BASE_URL,
                max_retries=0,
                timeout=10,
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=AsyncMock(side_effect=failure))
                ),
            )
            with self.assertRaises(type(failure)):
                await complete_async(client, capability="vision", model="vision-first")
            self.assertEqual(client.chat.completions.create.await_count, 1)
            self.assertEqual(self.pool.states(), {})

    async def test_disabled_mode_preserves_fixed_model(self):
        """Functionality: Preserve explicitly disabled pool behavior and original model selection.
        Inputs: disabled switch with an unapproved preferred model. Outputs: one original request.
        Logic: No approval/state admission runs. Constraints: This is explicit baseline mode.
        """
        client, requests = self.sdk([(200, None)])
        with patch.dict(
            "os.environ",
            {"FREE_MODEL_POOL_ENABLED": "false", "RESUME_VISION_MODEL": "original-fixed"},
        ):
            await self.review(client)
        self.assertEqual([r["model"] for r in requests], ["original-fixed"])
        self.assertEqual(self.pool.states(), {})

    async def test_invalid_approval_fails_before_dispatch(self):
        """Functionality: Refuse credential mismatch and invalid approval/storage configuration.
        Inputs: changed key, corrupt manifest, missing state parent. Outputs: preflight failures.
        Logic: Introduce isolated defects before selection. Constraints: No fallback or inference.
        """
        with self.assertRaises(PoolError):
            FreePool("different-key", BASE_URL)
        with patch.dict("os.environ", {"FREE_MODEL_POOL_STATE": str(self.root / "missing/db")}):
            with self.assertRaises(__import__("sqlite3").OperationalError):
                FreePool(KEY, BASE_URL)
        self.manifest.write_text("{bad", encoding="utf-8")
        client, requests = self.sdk([])
        with self.assertRaises(ValueError):
            await self.review(client)
        self.assertEqual(requests, [])

    async def test_shared_deadline_limits_rotation(self):
        """Functionality: Stop candidate advancement when the original request budget is spent.
        Inputs: deterministic clock and one quota error. Outputs: timeout before second request.
        Logic: Preparation at zero, first request at one, next at eleven with a ten-second budget.
        Constraints: Uncalled candidates must not acquire a terminal state.
        """
        client, requests = self.sdk([(403, "AllocationQuota.FreeTierOnly")])
        with patch("app.providers.free_pool.perf_counter", Mock(side_effect=[0, 1, 11])):
            with self.assertRaises(TimeoutError):
                await self.review(client)
        self.assertEqual(len(requests), 1)
        self.assertNotIn("vision-next", self.pool.states())

    def test_late_success_preserves_terminal_state(self):
        """Functionality: Verify monotonic blocking across independent SQLite connections.
        Inputs: exhausted model followed by eight concurrent late successes. Outputs: retained block
        and accumulated usage. Logic: Each writer loads its own pool. Constraints: No vendor calls.
        """
        self.pool.record("vision-first", "exhausted")
        response = SimpleNamespace(usage=SimpleNamespace(total_tokens=2))
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    FreePool(KEY, BASE_URL).record, "vision-first", "available", response
                )
                for _ in range(8)
            ]
            for future in futures:
                future.result()
        state = self.pool.states()["vision-first"]
        self.assertEqual((state["status"], state["observed_tokens"]), ("exhausted", 16))

    async def test_provider_and_request_guards(self):
        """Functionality: Reject providers/options unsupported by the established pool policy.
        Inputs: foreign provider, invalid switch, tools, stream, search, SDK retries and infinity.
        Outputs: preflight PoolError and zero requests. Logic: Validate without changing options.
        Constraints: No stripping options or silently disabling admission to make requests succeed.
        """
        for environment in [
            {"RESUME_VISION_PROVIDER": "openai"},
            {"FREE_MODEL_POOL_ENABLED": "invalid"},
        ]:
            with patch.dict("os.environ", environment), self.assertRaises(PoolError):
                ResumeVision()
        for options in [
            {"stream": True},
            {"tools": [{"type": "function"}]},
            {"extra_body": {"enable_search": True}},
            {"timeout": float("inf")},
        ]:
            client, requests = self.sdk([])
            async with client:
                with self.assertRaises(PoolError):
                    await complete_async(
                        client, capability="vision", model="vision-first", **options
                    )
            self.assertEqual(requests, [])
        client, requests = self.sdk([])
        client.max_retries = 1
        async with client:
            with self.assertRaises(PoolError):
                await complete_async(client, capability="vision", model="vision-first")
        self.assertEqual(requests, [])
