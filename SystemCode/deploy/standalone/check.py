"""Responsibilities: Verify a real standalone deployment without calling paid model APIs.
Implementation: Exercise PDF isolation, database access, authenticated HTTP/WebSocket transport and
one Celery probe. Temporary accounts and sessions are removed even when a check fails.
Related Modules: deploy.py invokes these checks; Compose uses readiness as its web health check.
Declaration Index:
- setup: Initialize Django from the deployment environment.
- configuration: Check Django, model and voice configuration without sending requests.
- request: Read one bounded internal HTTP response with the configured Host and protocol.
- readiness: Check the database and the real ASGI login page.
- sandbox: Verify the existing sandbox's isolation evidence and render a synthetic PDF.
- application: Verify account enforcement, authenticated endpoints, WebSocket and the real queue.
- public: Verify the public proxy's trusted TLS certificate and login form.
- main: Dispatch one check and report only sanitized failure context.
Variable Index:
None
"""

import asyncio
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from socket import create_connection
from uuid import uuid4


def setup():
    """Functionality: Initialize the deployed Django application.
    Inputs: Existing process settings and the source tree location. Outputs: None.
    Logic: Add only known source directories before Django setup; no developer .env is copied.
    Constraints: Missing or invalid settings fail without replacing configuration.
    """
    import django

    sys.path[:0] = [
        str(Path(__file__).resolve().parents[2] / "backend"),
        str(Path(__file__).resolve().parents[2]),
    ]
    django.setup()


def request(path, cookie="", *, port=8765):
    """Functionality: Fetch a bounded response from the real loopback ASGI server.
    Inputs: Application path, optional session cookie and loopback ASGI/proxy port. Outputs: bytes.
    Logic: Supply the configured Host and protocol exactly as the trusted proxy does.
    Constraints: One request with a 10-second timeout; credentials and response text are not logged.
    """
    host = os.environ["DJANGO_ALLOWED_HOSTS"].split(",")[0]
    headers = {
        "Host": host,
        "X-Forwarded-Proto": "https" if os.environ["DEPLOY_MODE"] == "server" else "http",
    }
    if cookie:
        headers["Cookie"] = cookie
    with urllib.request.urlopen(
        urllib.request.Request(f"http://127.0.0.1:{port}" + path, headers=headers), timeout=10
    ) as response:
        if response.status != 200:
            raise RuntimeError("Unexpected HTTP response status")
        return response.read(16 * 1024 * 1024)


def configuration():
    """Functionality: Validate framework/provider contracts before interrupting services.
    Inputs: Initialized Django and the saved application environment. Outputs: None.
    Logic: Run Django checks, construct/close the model client and validate the speech settings.
    Constraints: No provider request is sent; this cannot validate key permissions or quotas.
    """
    from django.core.management import call_command
    from interviews.agent_provider import BackendLLM
    from interviews.speech.service import SpeechConfig

    call_command("check")
    model = BackendLLM()
    model.client.close()
    SpeechConfig.load()


def readiness():
    """Functionality: Check database connectivity and login-page readiness.
    Inputs: Initialized Django environment. Outputs: None; failures propagate.
    Logic: Execute SELECT 1 and check the actual login form rather than TCP availability alone.
    Constraints: No accounts, sessions, model calls or automatic request retries are created.
    """
    from django.db import connection

    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        if cursor.fetchone() != (1,):
            raise RuntimeError("Database readiness query failed")
    if b'name="username"' not in request("/login/"):
        raise RuntimeError("ASGI login form is missing")


async def sandbox():
    """Functionality: Verify actual nested PDF isolation and parser installation.
    Inputs: Unchanged runtime paths/limits in the worker environment. Outputs: None.
    Logic: Require network, process, environment and memory isolation evidence; create one text
    synthetic page in memory and parse/render it through the same production sandbox.
    Constraints: No real resume/model input or in-process fallback; failures stop deployment.
    """
    from interviews.pdf_sandbox import parse_pdf, run_sandbox
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    print("Checking PDF namespace, seccomp and memory isolation", flush=True)
    evidence = await run_sandbox(b"", "probe")
    required = (
        "host_hidden",
        "secrets_absent",
        "socket_blocked",
        "fork_blocked",
        "memory_enforced",
    )
    if any(evidence.get(key) is not True for key in required):
        raise RuntimeError("PDF sandbox isolation probe failed")
    writer = PdfWriter()
    page = writer.add_blank_page(width=144, height=144)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 10 100 Td (Installation probe) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    buffer = io.BytesIO()
    writer.write(buffer)
    print("Checking PDF parser and image rendering", flush=True)
    pages = await parse_pdf(buffer.getvalue())
    if (
        len(pages) != 1
        or "Installation probe" not in pages[0].text
        or not pages[0].image_png.startswith(b"\x89PNG")
    ):
        raise RuntimeError("PDF sandbox parser/render probe failed")


def application():
    """Functionality: Verify core services and authentication without invoking interview models.
    Inputs: Initialized Django/Redis/Celery and a running ASGI service. Outputs: None.
    Logic: Reject an anonymous API request, create a temporary unusable-password user/session,
    check authenticated routes and WS hello through Caddy in local mode (ASGI in server mode),
    then send exactly one no-model Celery probe. Public server TLS is checked separately.
    Constraints: Queue polling observes one task rather than resending; all temporary records and
    Redis keys are removed in finally blocks. Provider access and scoring quality are not verified.
    """
    from config.celery import app
    from django.contrib.auth import (
        BACKEND_SESSION_KEY,
        HASH_SESSION_KEY,
        SESSION_KEY,
        get_user_model,
    )
    from django.contrib.sessions.backends.db import SessionStore
    from redis import Redis
    from websockets.sync.client import connect

    print("Checking database, login page and anonymous API rejection", flush=True)
    readiness()
    port = 80 if os.environ["DEPLOY_MODE"] == "local" else 8765
    try:
        request("/api/health/", port=port)
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise
    else:
        raise RuntimeError("Anonymous API access was unexpectedly allowed")
    user = get_user_model().objects.create_user(username="install-probe-" + uuid4().hex)
    session = SessionStore()
    try:
        print("Checking authenticated pages and WebSocket handshake", flush=True)
        session[SESSION_KEY] = str(user.pk)
        session[BACKEND_SESSION_KEY] = "django.contrib.auth.backends.ModelBackend"
        session[HASH_SESSION_KEY] = user.get_session_auth_hash()
        session.save()
        cookie = "sessionid=" + session.session_key
        health = json.loads(request("/api/health/", cookie, port=port))
        if health != {"status": "ok", "database": "postgresql"}:
            raise RuntimeError("Authenticated PostgreSQL health check failed")
        for path in ("/agent/", "/resumes/", "/stream-demo/pixel-player.js"):
            if not request(path, cookie, port=port):
                raise RuntimeError("Authenticated application resource is empty")
        host = os.environ["DJANGO_ALLOWED_HOSTS"].split(",")[0]
        scheme = "https" if os.environ["DEPLOY_MODE"] == "server" else "http"
        with (
            create_connection(("127.0.0.1", port), timeout=10) as transport,
            connect(
                f"ws://{host}/ws/agent/",
                sock=transport,
                origin=f"{scheme}://{host}",
                proxy=None,
                open_timeout=10,
                additional_headers={"Cookie": cookie, "X-Forwarded-Proto": scheme},
            ) as websocket,
        ):
            if json.loads(websocket.recv(timeout=10)).get("type") != "hello":
                raise RuntimeError("Authenticated WebSocket handshake failed")
    finally:
        session.delete()
        user.delete()
    redis = Redis.from_url(os.environ["PDF_TASK_REDIS_URL"], socket_timeout=5)
    probe = uuid4().hex
    key = f"ai-interviewer:probe:{probe}"
    try:
        print("Checking one real Celery task (no model call)", flush=True)
        app.send_task("interviews.worker_probe", args=[probe], retry=False, expires=20)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if redis.get(key) == b"ok":
                break
            time.sleep(0.25)
        else:
            raise TimeoutError("Celery worker did not consume the deployment probe")
    finally:
        redis.delete(key)
        redis.close()


def public(url):
    """Functionality: Verify the externally addressed proxy with normal TLS certificate validation.
    Inputs: Configured HTTPS origin. Outputs: None.
    Logic: Request its login page once; the system CA store verifies hostname and certificate.
    Constraints: Does not bypass certificate validation or retry ACME/DNS failures.
    """
    with urllib.request.urlopen(url + "/login/", timeout=30) as response:
        if response.status != 200 or b'name="username"' not in response.read(1024 * 1024):
            raise RuntimeError("Public HTTPS login verification failed")


def main():
    """Functionality: Dispatch one deployment check and return a meaningful exit status.
    Inputs: CLI check name and, for public, origin. Outputs: 0 on success, 1 on a visible failure.
    Logic: Initialize Django for internal checks; log phase, exception class and source location.
    Constraints: Exception messages can contain connection strings and are deliberately not printed.
    """
    mode = sys.argv[1]
    try:
        if mode == "public":
            public(sys.argv[2])
        else:
            setup()
            if mode == "sandbox":
                asyncio.run(sandbox())
            elif mode == "application":
                application()
            elif mode == "readiness":
                readiness()
            elif mode == "configuration":
                configuration()
            else:
                raise ValueError("Unknown check")
    except Exception as error:
        trace = error.__traceback__
        while trace.tb_next is not None:
            trace = trace.tb_next
        location = f"{Path(trace.tb_frame.f_code.co_filename).name}:{trace.tb_lineno}"
        print(
            f"FAIL {mode}: {type(error).__name__} at {location}; inspect logs/configuration",
            file=sys.stderr,
        )
        return 1
    if mode != "readiness":
        print(f"PASS {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
