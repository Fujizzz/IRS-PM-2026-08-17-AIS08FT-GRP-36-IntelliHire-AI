"""Responsibilities: Route calls only through explicitly approved free-quota models.
Implementation: Bind a private manifest to one key/endpoint and persist model observations in
SQLite across ASGI/Celery processes. Only explicit quota/model-access errors advance the pool;
network, authentication, rate-limit, parsing and cancellation failures propagate unchanged.
Related Modules: backend.interviews.resume_vision uses complete_async for PDF page review.
The existing private manifest/state schema also contains text and speech entries; this restoration
adds only the asynchronous chat route consumed by resume vision, without changing other adapters.
Declaration Index:
- PoolError: Safe failure when free-only admission or candidates are unavailable.
- enabled: Validate the explicit feature switch.
- require_provider: Refuse unguarded providers while free-only mode is enabled.
- identity: Bind model observations and approval to a key and regional endpoint.
- failure_reason: Recognize only documented, model-specific terminal API codes.
- FreePool: Validate approval and manage durable, process-shared observations.
- FreePool.__init__: Load and validate the manifest and create the state schema.
- FreePool.connect: Open one short-lived transactional SQLite connection.
- FreePool.states: Return persisted metadata for the current key/endpoint.
- FreePool.candidates: Order approved compatible models, preferring the configured original.
- FreePool.usable: Recheck a candidate immediately before sending its request.
- FreePool.record: Persist terminal or successful observations without resurrecting blocks.
- prepare: Validate a bounded chat request and construct its routing context.
- arguments: Preserve request parameters while bounding the next attempt's remaining timeout.
- complete_async: Perform asynchronous quota-aware model selection.
Variable Index:
- logger: Metadata-only model/state/usage diagnostics; never log keys or request/response bodies.
- BASE_URL: The Beijing compatible endpoint supported by the verified free-quota policy.
- CAPABILITIES: Explicit protocol groups; catalog visibility alone never implies compatibility.
- TERMINAL_CODES: Exact provider codes eligible for pool advancement.
Configuration Notes:
FREE_MODEL_POOL_ENABLED explicitly enables routing; false preserves the fixed-model request.
FREE_MODEL_POOL_MANIFEST is the absolute private approval path, bound to the key and endpoint.
FREE_MODEL_POOL_STATE is the absolute shared SQLite path; observations survive process restarts.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from time import perf_counter, time

from openai import APIStatusError

logger = logging.getLogger(__name__)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CAPABILITIES = {"text", "vision", "speech_tts", "speech_asr"}
TERMINAL_CODES = {
    "AllocationQuota.FreeTierOnly": "exhausted",
    "FreeTierOnly": "exhausted",
    "model_not_found": "unavailable",
    "ModelNotFound": "unavailable",
    "Endpoint.AccessDenied": "unavailable",
    "App.AccessDenied": "unavailable",
}


class PoolError(RuntimeError):
    """Functionality: Report admission failure without sensitive provider details.
    Logic: Carry a stable public error string. Constraints: No implicit paid or original-model path.
    """


def enabled():
    """Functionality: Resolve the explicit pool switch. Inputs: process environment.
    Outputs: bool. Logic: accept true/false only, default false for existing offline experiments.
    Constraints: Invalid configuration fails instead of silently disabling the free-only guard.
    """
    value = os.getenv("FREE_MODEL_POOL_ENABLED", "false").strip().lower()
    if value not in {"true", "false"}:
        raise PoolError("FREE_MODEL_POOL_ENABLED must be true or false")
    return value == "true"


def require_provider(provider):
    """Functionality: Reject other billing paths in free-only mode. Inputs: provider identifier.
    Outputs: None. Logic: only verified DashScope is supported. Constraints: No network or mutation.
    """
    if enabled() and provider != "dashscope":
        raise PoolError("free_pool_unsupported_provider")


def identity(key, base_url):
    """Functionality: Bind approval/state to credentials without storing their plaintext.
    Inputs: key and endpoint. Outputs: SHA256 namespace. Logic: hash endpoint, separator and key.
    Constraints: Requires nonempty credentials and exact Beijing HTTPS origin/path; no URL queries.
    """
    if not key or str(base_url).rstrip("/") != BASE_URL:
        raise PoolError("free_pool_requires_beijing_key_and_endpoint")
    return hashlib.sha256((BASE_URL + "\0" + key).encode()).hexdigest()


def failure_reason(error):
    """Functionality: Distinguish terminal model errors from unrelated failures.
    Inputs: SDK exception. Outputs: exhausted/unavailable or None. Logic: require both compatible
    HTTP status and an exact allowlisted code; never inspect arbitrary message substrings.
    Constraints: 401, 429, timeouts, account arrears and unknown 403s never rotate or blacklist.
    """
    if not isinstance(error, APIStatusError) or error.status_code not in {400, 403, 404}:
        return None
    body = error.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    code = body.get("code") if isinstance(body, dict) else None
    return TERMINAL_CODES.get(code) if isinstance(code, str) else None


class FreePool:
    """Functionality: Maintain a key-scoped model admission list and durable unavailable list.
    Logic: Only manifest entries confirmed free-only by the operator may be invoked. SQLite
    transactions serialize updates across processes, but never span network requests.
    Constraints: Metadata contains no prompts, keys or user data. Success cannot clear a block.
    """

    def __init__(self, key, base_url):
        """Functionality: Load strict approval and initialize persistent state.
        Inputs: deployed key/endpoint plus FREE_MODEL_POOL_MANIFEST/STATE absolute paths.
        Outputs: initialized pool. Logic: verify version, key fingerprint, unique model identifiers,
        free_only=True and explicit capabilities. Constraints: missing/corrupt files or SQLite
        errors fail closed; parent directories must already exist with deployment permissions.
        """
        self.namespace = identity(key, base_url)
        manifest_path = Path(os.environ.get("FREE_MODEL_POOL_MANIFEST", ""))
        self.state_path = Path(os.environ.get("FREE_MODEL_POOL_STATE", ""))
        if not manifest_path.is_absolute() or not self.state_path.is_absolute():
            raise PoolError("free_pool_requires_absolute_manifest_and_state_paths")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("version") != 1
            or manifest.get("identity") != self.namespace
            or not isinstance(manifest.get("models"), list)
            or not manifest["models"]
        ):
            raise PoolError("free_pool_invalid_manifest_or_key_binding")
        self.models = {}
        for entry in manifest["models"]:
            if not isinstance(entry, dict):
                raise PoolError("free_pool_invalid_model_approval")
            name = entry.get("id")
            capabilities = entry.get("capabilities")
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", name)
                or name in self.models
                or entry.get("free_only") is not True
                or not isinstance(capabilities, list)
                or not capabilities
                or any(not isinstance(c, str) or c not in CAPABILITIES for c in capabilities)
            ):
                raise PoolError("free_pool_invalid_model_approval")
            self.models[name] = frozenset(capabilities)
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS model_state (namespace TEXT NOT NULL, "
                "model TEXT NOT NULL, status TEXT NOT NULL, observed_at REAL NOT NULL, "
                "observed_tokens INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(namespace, model))"
            )

    def connect(self):
        """Functionality: Open a bounded SQLite connection. Inputs: configured state path.
        Outputs: connection owned by caller. Logic: wait at most one second for database locks.
        Constraints: No memory fallback or database recreation on errors; callers close promptly.
        """
        return sqlite3.connect(self.state_path, timeout=1)

    def states(self):
        """Functionality: Read current observations. Inputs: instance namespace and database.
        Outputs: model-to-metadata mapping. Logic: key/endpoint-scoped SELECT, short connection.
        Constraints: observed_tokens is local successful usage, never authoritative remaining quota.
        """
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT model,status,observed_at,observed_tokens "
                "FROM model_state WHERE namespace=?",
                (self.namespace,),
            ).fetchall()
        return {
            model: {"status": status, "observed_at": at, "observed_tokens": tokens}
            for model, status, at, tokens in rows
        }

    def candidates(self, preferred, capability):
        """Functionality: Select compatible approved models in deterministic order.
        Inputs: original model and required capability. Outputs: finite unique candidate list.
        Logic: prefer the original when approved; then manifest order; exclude persisted blocks.
        Constraints: An unapproved preferred model is never called, even when the list is empty.
        """
        states = self.states()
        names = [name for name, caps in self.models.items() if capability in caps]
        if preferred in names:
            names.remove(preferred)
            names.insert(0, preferred)
        return [
            name
            for name in names
            if states.get(name, {}).get("status") not in {"exhausted", "unavailable"}
        ]

    def usable(self, model):
        """Functionality: Recheck a model before dispatch. Inputs: approved model identifier.
        Outputs: bool. Logic: consult latest process-shared terminal state.
        Constraints: Requests already in flight cannot be recalled; vendor free-only is the billing
        boundary. A concurrent success must not remove another process's terminal observation.
        """
        return self.states().get(model, {}).get("status") not in {"exhausted", "unavailable"}

    def record(self, model, status, response=None):
        """Functionality: Atomically retain terminal states and observed successful token usage.
        Inputs: approved model, status, optional SDK response. Outputs: None; writes metadata/logs.
        Logic: UPSERT adds successful usage but preserves existing terminal status on late success.
        Constraints: No TTL reset, raw response storage, or silent state-write failures. Missing
        usage is zero observed tokens and does not imply the request cost zero tokens.
        """
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", 0)
        tokens = tokens if type(tokens) is int and tokens >= 0 else 0
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "INSERT INTO model_state VALUES(?,?,?,?,?) ON CONFLICT(namespace,model) "
                "DO UPDATE SET status=CASE WHEN model_state.status IN ('exhausted','unavailable') "
                "AND excluded.status='available' THEN model_state.status ELSE excluded.status END, "
                "observed_at=excluded.observed_at, "
                "observed_tokens=model_state.observed_tokens+excluded.observed_tokens",
                (self.namespace, model, status, time(), tokens),
            )
        logger.log(
            logging.INFO if status == "available" else logging.WARNING,
            "Free model pool model=%s status=%s observed_tokens=%s",
            model,
            status,
            tokens,
        )


def prepare(client, capability, kwargs):
    """Functionality: Validate admission before any inference and share one timeout budget.
    Inputs: SDK client, capability, original call kwargs. Outputs: pool, candidates, deadline.
    Logic: reject streaming/tools/search, require zero SDK retries and finite request timeout;
    validate key-bound approval and select compatible models. Constraints: manifest loading and
    state reads consume the same budget; only text/vision chat-completions are routed here.
    """
    started = perf_counter()
    extra = kwargs.get("extra_body") or {}
    if (
        capability not in {"text", "vision"}
        or kwargs.get("stream")
        or kwargs.get("tools")
        or kwargs.get("functions")
        or set(extra) - {"enable_thinking"}
        or client.max_retries != 0
    ):
        raise PoolError("free_pool_unsupported_request_options")
    timeout = kwargs.get("timeout", client.timeout)
    if not isinstance(timeout, (int, float)):
        timeout = timeout.read
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise PoolError("free_pool_requires_finite_timeout")
    pool = FreePool(client.api_key, str(client.base_url))
    candidates = pool.candidates(kwargs["model"], capability)
    return pool, candidates, started + timeout


def arguments(kwargs, model, deadline):
    """Functionality: Preserve inference options while limiting each attempt to remaining time.
    Inputs: original kwargs, selected model, shared monotonic deadline. Outputs: copied kwargs.
    Logic: replace only model and timeout. Constraints: elapsed budget raises before dispatch;
    complete_async additionally bounds the network await by the remaining wall-clock budget.
    """
    remaining = deadline - perf_counter()
    if remaining <= 0:
        raise TimeoutError("free_pool_deadline_elapsed")
    return {**kwargs, "model": model, "timeout": remaining}


async def complete_async(client, *, capability="text", **kwargs):
    """Functionality: Execute one asynchronous request through the explicitly enabled pool.
    Inputs: SDK client, required capability, and original completion keyword arguments.
    Outputs: SDK response, or the original failure; an empty pool raises PoolError.
    Logic: Disabled mode delegates unchanged. Enabled mode uses approved compatible candidates,
    each at most once, and advances only after allowlisted terminal model errors. Disk operations
    use worker threads; a shared remaining wall-time budget bounds network awaits.
    Constraints: No same-model retry, prompt/schema modification, or paid escape path. Cancellation,
    timeouts, authentication, rate limits and unknown errors propagate without rotation. SQLite
    writes already dispatched to a worker may finish after caller cancellation.
    """
    if not enabled():
        return await client.chat.completions.create(**kwargs)
    pool, candidates, deadline = await asyncio.to_thread(prepare, client, capability, kwargs)
    for model in candidates:
        if not await asyncio.to_thread(pool.usable, model):
            continue
        logger.info(
            "Free model pool dispatch requested=%s selected=%s capability=%s",
            kwargs["model"],
            model,
            capability,
        )
        options = arguments(kwargs, model, deadline)
        try:
            async with asyncio.timeout(options["timeout"]):
                response = await client.chat.completions.create(**options)
        except APIStatusError as error:
            reason = failure_reason(error)
            if reason is None:
                raise
            await asyncio.to_thread(pool.record, model, reason)
            continue
        await asyncio.to_thread(pool.record, model, "available", response)
        return response
    raise PoolError("free_pool_no_available_" + capability + "_models")
