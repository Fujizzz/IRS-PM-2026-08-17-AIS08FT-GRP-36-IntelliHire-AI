#!/usr/bin/env python3
"""Responsibilities: Provide an end-user local/server deployment command without Python packages.
Implementation: Persist private configuration, then build and operate the isolated Compose stack;
updates back up PostgreSQL before stopping services and applying migrations. Failures stay visible.
Related Modules: SystemCode/deploy/standalone contains the image, Compose services and checks.
Declaration Index:
- read_environment: Read the documented literal KEY=value input format without interpolation.
- write_private: Exclusively create a UTF-8 file with restrictive permissions.
- secure_directory: Restrict deployment state to the current user and, on Windows, SYSTEM.
- deployment_lock: Prevent concurrent deployment commands using an OS-owned file lock.
- ask: Obtain one configuration value without echoing secrets.
- validate_configuration: Reject missing model/voice inputs and infrastructure overrides.
- configure: Create or reuse persistent deployment configuration without overwriting secrets.
- run: Execute a named deployment stage without printing environment contents.
- compose_command: Build fixed Compose arguments with an explicit project and environment file.
- check_docker: Require a reachable Linux Docker engine and Compose with raw env-file support.
- backup: Save a PostgreSQL custom-format dump without passing its bytes through text decoding.
- deploy: Build, preflight, back up, migrate, start, and verify the configured application.
- main: Parse the public CLI and report actionable failures with nonzero exit status.
Variable Index:
- ROOT: Repository root resolved independently of the caller's working directory.
- ASSETS: Versioned standalone deployment resources.
- PROVIDERS: Supported interview model providers, matching BackendLLM.
- OWNED_KEYS: Infrastructure variables that users may not supply through a model configuration file.
- logger: Stage-only logger; credentials and model request bodies are never logged here.
"""

import argparse
import contextlib
import csv
import getpass
import json
import logging
import os
import re
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "SystemCode/deploy/standalone"
PROVIDERS = {"openai", "dashscope", "deepseek"}
OWNED_KEYS = {
    "DJANGO_SETTINGS_MODULE",
    "DJANGO_SECRET_KEY",
    "DJANGO_ALLOWED_HOSTS",
    "DEPLOY_MODE",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "CELERY_BROKER_URL",
    "PDF_TASK_REDIS_URL",
    "PDF_TASK_EXECUTION",
    "PDF_SANDBOX_RUNTIME",
    "PDF_SANDBOX_DISTRO",
    "SERVICE_CAPACITY_DIR",
    "RECOMMENDATION_JOB_CATALOG",
}
logger = logging.getLogger("intellihire.deploy")


def read_environment(path):
    """Functionality: Read a literal, single-line KEY=value file.
    Inputs: A UTF-8 path; blank/comment lines are ignored. Outputs: A string dictionary.
    Logic: Split at the first equals sign and validate unique shell-style names.
    Constraints: No shell evaluation, quote removal, expansion or multiline values.
    Error messages omit values.
    """
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or key in values:
            raise ValueError(f"Invalid or duplicate configuration name at {path.name}:{number}")
        if "\x00" in value:
            raise ValueError(f"Invalid configuration value at {path.name}:{number}")
        values[key] = value
    return values


def write_private(path, content):
    """Functionality: Create a private configuration or receipt without replacing an existing file.
    Inputs: Destination path and text. Outputs: None; the file is persisted with LF line endings.
    Logic: O_EXCL prevents accidental secret replacement; Unix mode is 0600 at creation.
    Constraints: The parent must already have its private directory permissions applied.
    """
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def secure_directory(path):
    """Functionality: Create and protect deployment state before writing credentials.
    Inputs: An absolute directory path. Outputs: None; permission failures propagate.
    Logic: Use owner-only Unix mode, or Windows SID-based inherited ACLs for the owner and SYSTEM.
    Constraints: Windows chmod alone cannot protect secrets; no broad Users/Everyone grant is added.
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "nt":
        identity = subprocess.check_output(["whoami", "/user", "/fo", "csv", "/nh"], text=True)
        sid = next(csv.reader(identity.strip().splitlines()))[1]
        subprocess.run(
            [
                "icacls",
                str(path),
                "/inheritance:r",
                "/grant:r",
                f"*{sid}:(OI)(CI)F",
                "*S-1-5-18:(OI)(CI)F",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        path.chmod(0o700)


@contextlib.contextmanager
def deployment_lock(state):
    """Functionality: Serialize commands against one deployment without stale PID lock files.
    Inputs: Private state directory. Outputs: A context held for the entire operation.
    Logic: Lock one byte on Windows or flock on Unix; process exit also releases the OS lock.
    Constraints: A busy deployment fails without modifying its services or configuration.
    """
    with (state / "operation.lock").open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                "Another deployment command is using this state directory"
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def ask(label, default="", *, secret=False):
    """Functionality: Read one interactive setup value, optionally through hidden input.
    Inputs: Display label, nonsecret default, and secret flag. Outputs: A single stripped string.
    Logic: Require a terminal; noninteractive users must supply --config instead.
    Constraints: Secrets never become command-line arguments or log messages.
    """
    if not sys.stdin.isatty():
        raise ValueError("First setup needs a terminal or --config PATH; see DEPLOYMENT.md")
    prompt = label + (f" [{default}]" if default else "") + ": "
    value = (getpass.getpass(prompt) if secret else input(prompt)).strip() or default
    if not value or any(character in value for character in "\r\n\x00"):
        raise ValueError(f"A single-line value is required for {label}")
    return value


def validate_configuration(values):
    """Functionality: Validate the model and speech contract of the existing interview webpage.
    Inputs: Literal environment entries. Outputs: None; invalid names/requirements raise ValueError.
    Logic: Require the selected provider's model/key plus enabled DashScope voice and an explicit
    matching region. Infrastructure settings are owned by the deployment command.
    Constraints: Checks presence, not provider access/quota; no paid model request is made.
    """
    if forbidden := OWNED_KEYS.intersection(values):
        raise ValueError(
            "Deployment owns these configuration names: " + ", ".join(sorted(forbidden))
        )
    provider = values.get("LLM_PROVIDER", "").strip()
    if provider not in PROVIDERS:
        raise ValueError("LLM_PROVIDER must be openai, dashscope or deepseek")
    required = [f"{provider.upper()}_API_KEY", f"{provider.upper()}_MODEL", "DASHSCOPE_API_KEY"]
    if any(not values.get(key, "").strip() for key in required):
        raise ValueError(
            "Configure the interview provider model/key and DASHSCOPE_API_KEY for voice"
        )
    if values.get("SPEECH_ENABLED") != "true":
        raise ValueError("The current interview page needs SPEECH_ENABLED=true for voice answers")
    if values.get("SPEECH_REGION") not in {"beijing", "singapore"}:
        raise ValueError("SPEECH_REGION must explicitly match the key: beijing or singapore")
    if not values.get("DASHSCOPE_BASE_URL", "").startswith("https://"):
        raise ValueError("Set DASHSCOPE_BASE_URL to the HTTPS endpoint matching the DashScope key")


def configure(args, state):
    """Functionality: Create a persistent deployment or validate an existing deployment identity.
    Inputs: Parsed CLI arguments and private state directory. Outputs: Saved nonsecret metadata.
    Logic: Save model inputs, generated independent secrets, proxy config and Compose interpolation
    separately. Write deployment.json last as the configuration-complete marker.
    Constraints: Never read the developer .env, rotate stored secrets, or switch an existing mode.
    An interrupted first setup requires explicit operator repair of the partial directory.
    """
    state_text = state.as_posix()
    if any(character in state_text for character in "'\r\n"):
        raise ValueError("The state directory cannot contain apostrophes or newlines")
    if args.domain is not None:
        args.domain = args.domain.lower()
    marker = state / "deployment.json"
    if marker.exists():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if args.action in {"local", "server"} and args.action != metadata["mode"]:
            raise ValueError("Use another --state-dir for another deployment mode")
        for name in ("domain", "port", "name"):
            requested = getattr(args, name)
            if requested is not None and requested != metadata[name]:
                raise ValueError(f"Existing {name} differs; use another --state-dir")
        if args.config:
            raise ValueError("Existing setup is preserved; edit its app.env instead of --config")
        return metadata
    if args.action not in {"local", "server"}:
        raise ValueError("No saved deployment. Start with: python deploy.py local")
    if any(path.name != "operation.lock" for path in state.iterdir()):
        raise ValueError("Incomplete setup directory; inspect it before choosing a new --state-dir")
    domain = args.domain or ""
    if args.action == "server" and not re.fullmatch(
        r"(?=.{1,253}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
        r"[a-zA-Z][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]",
        domain,
    ):
        raise ValueError(
            "Server mode requires --domain interview.example.com (DNS, without scheme)"
        )
    if args.action == "local" and domain:
        raise ValueError("--domain is only valid for server mode")
    port = args.port if args.port is not None else 8765
    if not 1024 <= port <= 65535 or (args.action == "server" and args.port is not None):
        raise ValueError("--port is a local-mode port in 1024..65535; server uses 80/443")
    name = args.name or "intellihire"
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,40}", name):
        raise ValueError("--name must start with a lowercase letter and contain a-z, 0-9, _ or -")
    # These are explicit resource choices for a fresh core installation, not failure fallbacks.
    values = {
        "RAG_ENABLED": "false",
        "AVATAR_REMOTE_ENABLED": "false",
        "PRESENTATION_ENABLED": "false",
        "ANSWER_COMPLETION_GATE_PATH": "",
        "AI_SECURITY_ENABLED": "false",
        "QUESTION_QUALITY_MODE": "relaxed",
        "EVALUATION_MODE": "shadow",
        "EVALUATION_USE_WORKER": "true",
        "OPENAI_TEMPERATURE": "0",
        "SPEECH_ENABLED": "true",
        "SPEECH_TTS_MODEL": "qwen-audio-3.0-tts-flash",
        "SPEECH_TTS_VOICE": "loongeva_v3.6",
        "SPEECH_STT_MODEL": "qwen-audio-3.1-asr-flash-streaming",
    }
    if args.config:
        values.update(read_environment(Path(args.config).resolve()))
    else:
        provider = ask("Interview model provider (openai/dashscope/deepseek)")
        if provider not in PROVIDERS:
            raise ValueError("Unsupported interview provider")
        values["LLM_PROVIDER"] = provider
        values[f"{provider.upper()}_MODEL"] = ask("Interview model name")
        values[f"{provider.upper()}_API_KEY"] = ask("Interview API key", secret=True)
        if provider != "dashscope":
            values["DASHSCOPE_API_KEY"] = ask("DashScope voice API key", secret=True)
        region = ask("DashScope key region (beijing/singapore)")
        values["SPEECH_REGION"] = region
        endpoint = (
            "dashscope.aliyuncs.com" if region == "beijing" else "dashscope-intl.aliyuncs.com"
        )
        values["DASHSCOPE_BASE_URL"] = ask(
            "DashScope compatible API URL", f"https://{endpoint}/compatible-mode/v1"
        )
    validate_configuration(values)
    password, admin_password, redis_password = (secrets.token_hex(32) for _ in range(3))
    values.update(
        {
            "DJANGO_SETTINGS_MODULE": "config.standalone",
            "DEPLOY_MODE": args.action,
            "DJANGO_SECRET_KEY": secrets.token_urlsafe(64),
            "DJANGO_ALLOWED_HOSTS": domain if domain else "localhost,127.0.0.1",
            "POSTGRES_DB": "intellihire",
            "POSTGRES_USER": "intellihire",
            "POSTGRES_PASSWORD": password,
            "POSTGRES_HOST": "db",
            "POSTGRES_PORT": "5432",
            "CELERY_BROKER_URL": f"redis://:{redis_password}@redis:6379/0",
            "PDF_TASK_REDIS_URL": f"redis://:{redis_password}@redis:6379/1",
            "PDF_SANDBOX_RUNTIME": "/opt/pdf-sandbox",
            "SERVICE_CAPACITY_DIR": "/var/lib/intellihire/capacity",
            "RECOMMENDATION_JOB_CATALOG": (
                "/app/backend/interviews/recommendation/data/experience-jobs.json"
            ),
        }
    )
    write_private(state / "app.env", "".join(f"{key}={value}\n" for key, value in values.items()))
    write_private(
        state / "db.env", f"POSTGRES_PASSWORD={admin_password}\nAPP_PASSWORD={password}\n"
    )
    write_private(
        state / "redis.conf",
        "bind 0.0.0.0\nprotected-mode yes\nport 6379\n"
        f'requirepass {redis_password}\nsave ""\nappendonly no\n'
        "maxmemory 512mb\nmaxmemory-policy noeviction\n",
    )
    metadata = {"mode": args.action, "domain": domain, "port": port, "name": name}
    site = domain if domain else "http://:80"
    write_private(
        state / "Caddyfile",
        "{\n  admin off\n}\n" + site + " {\n"
        "  reverse_proxy 127.0.0.1:8765 {\n    flush_interval -1\n"
        "    header_up -X-Forwarded-For\n    header_up -Forwarded\n"
        "    header_up -Authorization\n    header_up X-Forwarded-Proto {scheme}\n  }\n}\n",
    )
    # Compose paths use forward slashes on Windows; single quotes suppress variable interpolation.
    write_private(
        state / "compose.env",
        f"STATE_DIR='{state_text}'\n"
        f"HTTP_BIND={'0.0.0.0' if domain else '127.0.0.1'}\n"
        f"HTTP_PORT={80 if domain else port}\n",
    )
    write_private(marker, json.dumps(metadata, indent=2) + "\n")
    logger.info("Private configuration saved; UE/RAG/offline completion assets are opt-in")
    return metadata


def run(label, arguments, **kwargs):
    """Functionality: Run one external deployment stage and retain its exit status.
    Inputs: Nonsecret stage label, argv and subprocess options. Outputs: CompletedProcess.
    Logic: Print the stage only, clear inherited Compose interpolation overrides, invoke without
    a shell, and propagate failure without retry. The selected Docker context remains unchanged.
    Constraints: Callers must not put credentials into argv or capture secret-producing commands.
    """
    logger.info("%s", label)
    kwargs.setdefault(
        "env",
        {
            key: value
            for key, value in os.environ.items()
            if key not in {"STATE_DIR", "HTTP_BIND", "HTTP_PORT"} and not key.startswith("COMPOSE_")
        },
    )
    return subprocess.run(arguments, check=True, **kwargs)


def compose_command(state, metadata):
    """Functionality: Construct an unambiguous Compose project command.
    Inputs: State directory and saved identity. Outputs: An argv prefix.
    Logic: Always select files explicitly; server mode adds public HTTPS port publication.
    Constraints: Paths may contain spaces; callers append argv elements, never shell fragments.
    """
    command = [
        "docker",
        "compose",
        "--project-name",
        metadata["name"],
        "--env-file",
        str(state / "compose.env"),
        "-f",
        str(ASSETS / "compose.yaml"),
    ]
    if metadata["mode"] == "server":
        command += ["-f", str(ASSETS / "server.yaml")]
    return command


def check_docker():
    """Functionality: Fail early when the documented container prerequisite is missing.
    Inputs: Docker CLI/context on PATH. Outputs: None.
    Logic: Check Compose >=2.30 and a reachable Linux engine; reject Windows-container engines.
    Constraints: Does not install system software or change the user's Docker context.
    """
    version = subprocess.check_output(
        ["docker", "compose", "version", "--short"], text=True
    ).strip()
    match = re.match(r"v?(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (2, 30):
        raise RuntimeError("Docker Compose 2.30+ is required; update Docker Desktop/Compose")
    system = subprocess.check_output(
        ["docker", "info", "--format", "{{.OSType}}"], text=True
    ).strip()
    if system != "linux":
        raise RuntimeError("Select the Linux containers engine in Docker Desktop")


def backup(command, state):
    """Functionality: Persist a custom-format database backup before a schema update.
    Inputs: Compose prefix and private state directory. Outputs: Absolute dump path.
    Logic: Stream pg_dump bytes directly into an exclusively created file; rename only on success.
    Constraints: No password in argv; failed partial dumps remain marked .partial for diagnosis.
    """
    directory = state / "backups"
    directory.mkdir(exist_ok=True, mode=0o700)
    path = directory / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".dump")
    partial = path.with_suffix(".partial")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        run(
            "Back up PostgreSQL",
            command + ["exec", "-T", "db", "pg_dump", "-U", "postgres", "-d", "intellihire", "-Fc"],
            stdout=stream,
        )
    if partial.stat().st_size == 0:
        raise RuntimeError("Database backup is empty; services have not been stopped")
    partial.rename(path)
    logger.info("Database backup saved: %s", path)
    return path


def deploy(command, state, metadata):
    """Functionality: Install/update the stack with explicit migration and verification boundaries.
    Inputs: Compose argv, private state and deployment metadata. Outputs: None on success.
    Logic: Build first; start DB/Redis; validate settings and sandbox; back up; stop application
    writers; migrate once; start and verify accounts, WebSocket and queue without model calls.
    Constraints: No automatic retry/rollback/restart. Failure after stopping leaves services stopped
    or their actual failing state; the operator inspects logs and backup before an explicit rerun.
    """
    run("Validate Compose configuration", command + ["config", "--quiet"])
    run("Build application with locked dependencies", command + ["build", "web"])
    run(
        "Start database and queue",
        command + ["up", "-d", "--wait", "--wait-timeout", "120", "db", "redis"],
    )
    run(
        "Check application configuration",
        command
        + [
            "run",
            "--rm",
            "--no-deps",
            "web",
            "python",
            "../deploy/standalone/check.py",
            "configuration",
        ],
    )
    run(
        "Check real PDF isolation",
        command
        + [
            "run",
            "--rm",
            "--no-deps",
            "worker",
            "python",
            "../deploy/standalone/check.py",
            "sandbox",
        ],
    )
    backup(command, state)
    run(
        "Stop application processes for migration",
        command + ["stop", "-t", "150", "proxy", "web", "worker", "evaluation"],
    )
    run(
        "Apply database migrations",
        command
        + ["run", "--rm", "--no-deps", "web", "python", "manage.py", "migrate", "--noinput"],
    )
    run(
        "Start application and workers",
        command
        + [
            "up",
            "-d",
            "--force-recreate",
            "--wait",
            "--wait-timeout",
            "120",
            "web",
            "worker",
            "evaluation",
            "proxy",
        ],
    )
    run(
        "Verify authenticated HTTP, WebSocket and Celery",
        command + ["exec", "-T", "web", "python", "../deploy/standalone/check.py", "application"],
    )
    url = (
        f"https://{metadata['domain']}"
        if metadata["domain"]
        else f"http://localhost:{metadata['port']}"
    )
    # Public TLS/DNS must succeed before server deployment is reported complete.
    if metadata["mode"] == "server":
        run(
            "Verify public HTTPS certificate and login page",
            command
            + ["exec", "-T", "web", "python", "../deploy/standalone/check.py", "public", url],
        )
    else:
        run(
            "Verify local HTTP proxy",
            command
            + [
                "exec",
                "-T",
                "web",
                "python",
                "../deploy/standalone/check.py",
                "public",
                "http://localhost:80",
            ],
        )
    logger.info("Application ready: %s/register/ (create your account).", url)
    logger.info(
        "Model credentials checked for presence only; complete an interview to verify providers."
    )


def main():
    """Functionality: Expose local/server deployment and non-destructive operational commands.
    Inputs: CLI arguments, interactive input on first setup and Docker access. Outputs: Exit code.
    Logic: Validate prerequisites, lock state, reuse configuration and dispatch one action.
    Constraints: stop preserves volumes; no command deletes data or changes the existing server.
    """
    parser = argparse.ArgumentParser(description="Deploy IntelliHire locally or on your server")
    parser.add_argument("action", choices=["local", "server", "status", "logs", "stop", "backup"])
    parser.add_argument(
        "--domain", help="Server DNS name; public ports 80/443 must reach this host"
    )
    parser.add_argument("--port", type=int, help="Local port (default: 8765)")
    parser.add_argument("--name", help="Compose project name for a new installation")
    parser.add_argument("--config", help="First setup: literal KEY=value model/voice configuration")
    parser.add_argument("--state-dir", type=Path, default=ROOT / ".deploy")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        check_docker()
        state = args.state_dir.resolve()
        secure_directory(state)
        with deployment_lock(state):
            metadata = configure(args, state)
            command = compose_command(state, metadata)
            if args.action in {"local", "server"}:
                deploy(command, state, metadata)
            elif args.action == "backup":
                backup(command, state)
            else:
                operation = {
                    "status": ["ps", "--all"],
                    "logs": ["logs", "--tail", "100", "-f"],
                    "stop": ["stop", "-t", "150"],
                }[args.action]
                run(args.action, command + operation)
    except FileNotFoundError as error:
        logger.error(
            "Required executable/file not found: %s. Check --config and Docker/Compose; "
            "see DEPLOYMENT.md",
            error.filename,
        )
        return 1
    except subprocess.CalledProcessError as error:
        logger.error(
            "Deployment stage failed (exit %s). No retry/rollback. "
            "Run: python deploy.py logs --state-dir %s",
            error.returncode,
            args.state_dir,
        )
        return 1
    except (ValueError, RuntimeError, OSError) as error:
        logger.error("Deployment stopped: %s", error)
        return 1
    except KeyboardInterrupt:
        logger.error("Command interrupted; running containers and persistent data are retained")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
