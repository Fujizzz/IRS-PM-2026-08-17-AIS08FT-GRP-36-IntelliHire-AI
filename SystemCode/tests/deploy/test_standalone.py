"""Responsibilities: Verify deployment configuration and service-interruption boundaries.
Implementation: Use temporary directories and mocked Docker commands; never contact real services.
Related Modules: Root deploy.py; real container checks live in deploy/standalone/check.py.
Declaration Index:
- installer: Import the standard-library deployment CLI without starting it.
- inputs: Build explicit synthetic model/voice inputs; they cannot authenticate to providers.
- arguments: Build first-install CLI options for a temporary installation.
- test_setup_is_private_stable_and_preserves_literal_secrets: Check persistence and literal values.
- test_rejects_incomplete_or_infrastructure_config: Check required inputs and override boundaries.
- test_server_validation_precedes_file_creation: Reject malformed origins before secrets exist.
- test_existing_identity_cannot_be_changed: Preserve deployment mode, ports and project identity.
- test_failed_preflight_does_not_stop_running_services: Keep running services if isolation fails.
- test_migration_failure_does_not_start_services_or_retry: Retain failure semantics after stopping.
- test_backup_is_binary_and_failure_keeps_partial: Preserve dump bytes and incomplete output.
- test_literal_file_rejects_duplicates_and_keeps_values: Avoid shell-style interpretation of keys.
- test_runtime_settings_preserve_server_security: Verify settings in isolated processes.
- test_parent_environment_cannot_publish_local_ports: Ignore ambient Compose interpolation values.
- test_server_hostname_is_normalized: Preserve equivalent DNS spelling across repeated setup.
Variable Index:
- ROOT: Git root containing the deployment CLI; determined from the test location.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def installer():
    """Functionality: Import only the installer definitions for isolated unit tests.
    Inputs: Repository layout. Outputs: A fresh module object.
    Logic: Execute the module with a non-main name. Constraints: No Docker/network is invoked.
    """
    spec = importlib.util.spec_from_file_location("standalone_installer", ROOT / "deploy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    """Functionality: Supply deliberately invalid-for-authentication test credentials.
    Inputs: None. Outputs: Complete configuration dictionary.
    Logic: Match the presence contract with obvious synthetic values.
    Constraints: These values are never used to call model services.
    """
    return {
        "LLM_PROVIDER": "dashscope",
        "DASHSCOPE_API_KEY": "synthetic-$key=#literal",
        "DASHSCOPE_MODEL": "synthetic-model",
        "SPEECH_ENABLED": "true",
        "SPEECH_REGION": "beijing",
        "DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }


def arguments(tmp_path, **overrides):
    """Functionality: Build setup arguments without interactive or external dependencies.
    Inputs: A pytest-owned temporary directory and explicit overrides. Outputs: argparse.Namespace.
    Logic: Persist only synthetic model values beside the private state directory.
    Constraints: Existing developer/production configuration is never read.
    """
    config = tmp_path / "models.env"
    config.write_text(
        "".join(f"{key}={value}\n" for key, value in inputs().items()), encoding="utf-8"
    )
    values = dict(action="local", domain=None, port=None, name=None, config=str(config))
    values.update(overrides)
    return argparse.Namespace(**values)


def test_setup_is_private_stable_and_preserves_literal_secrets(installer, tmp_path):
    """Functionality: Verify generated secrets are stable and separate from provider input.
    Inputs: Real temporary files, no subprocess mocks. Outputs: Assertions only.
    Logic: Configure twice; compare all saved files, literal values and Unix creation modes.
    Constraints: Windows ACL application is tested by integration, not by Unix mode assertions.
    """
    state = tmp_path / "state with spaces"
    state.mkdir()
    args = arguments(tmp_path)
    metadata = installer.configure(args, state)
    before = {path.name: path.read_bytes() for path in state.iterdir()}
    args.config = None
    assert installer.configure(args, state) == metadata
    assert before == {path.name: path.read_bytes() for path in state.iterdir()}
    app = installer.read_environment(state / "app.env")
    assert app["DASHSCOPE_API_KEY"] == inputs()["DASHSCOPE_API_KEY"]
    assert len(app["DJANGO_SECRET_KEY"]) >= 64
    assert app["POSTGRES_PASSWORD"] != app["DASHSCOPE_API_KEY"]
    assert "POSTGRES_PASSWORD" not in (state / "compose.env").read_text()
    assert app["EVALUATION_USE_WORKER"] == "true"
    assert app["AVATAR_REMOTE_ENABLED"] == app["RAG_ENABLED"] == "false"
    if os.name != "nt":
        assert (state / "app.env").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "change",
    [
        {"DASHSCOPE_API_KEY": ""},
        {"DASHSCOPE_MODEL": ""},
        {"SPEECH_ENABLED": "false"},
        {"SPEECH_REGION": "invalid"},
        {"LLM_PROVIDER": "unknown"},
        {"DJANGO_SECRET_KEY": "forbidden"},
        {"POSTGRES_HOST": "unrelated-database"},
    ],
)
def test_rejects_incomplete_or_infrastructure_config(installer, change):
    """Functionality: Reject incomplete providers and infrastructure override attempts.
    Inputs: Synthetic valid values with one explicit invalid change. Outputs: ValueError assertion.
    Logic: Invoke pure validation. Constraints: No inference, disk or database access occurs.
    """
    with pytest.raises(ValueError):
        installer.validate_configuration(inputs() | change)


@pytest.mark.parametrize(
    "domain", ["", "http://example.com", "127.0.0.1", "x.com\nadmin off", "a;bad.com", "x.com-"]
)
def test_server_validation_precedes_file_creation(installer, tmp_path, domain):
    """Functionality: Reject Caddy/DNS injection and ambiguous origins before writing configuration.
    Inputs: Invalid server names and an empty temporary state. Outputs: No configuration files.
    Logic: Attempt server setup and inspect the directory. Constraints: No DNS/network requests.
    """
    state = tmp_path / "state"
    state.mkdir()
    with pytest.raises(ValueError):
        installer.configure(arguments(tmp_path, action="server", domain=domain), state)
    assert list(state.iterdir()) == []


def test_existing_identity_cannot_be_changed(installer, tmp_path):
    """Functionality: Prevent a repeated command from changing the exposure of a saved deployment.
    Inputs: An existing local configuration. Outputs: Explicit mode/port/name errors.
    Logic: Attempt each identity change separately. Constraints: Secrets and volumes are untouched.
    """
    state = tmp_path / "state"
    state.mkdir()
    installer.configure(arguments(tmp_path), state)
    for override in ({"action": "server"}, {"port": 9000}, {"name": "different"}):
        with pytest.raises(ValueError):
            installer.configure(arguments(tmp_path, config=None, **override), state)


def test_failed_preflight_does_not_stop_running_services(installer, tmp_path, monkeypatch):
    """Functionality: Verify failed PDF isolation blocks deployment before service interruption.
    Inputs: Mocked Docker stages with a sandbox failure. Outputs: No backup/stop/migrate call.
    Logic: Inspect ordered calls. Constraints: Verifies orchestration, not actual isolation.
    """
    stages = Mock(side_effect=[None, None, None, None, subprocess.CalledProcessError(1, "probe")])
    monkeypatch.setattr(installer, "run", stages)
    save = Mock()
    monkeypatch.setattr(installer, "backup", save)
    with pytest.raises(subprocess.CalledProcessError):
        installer.deploy(
            ["docker", "compose"], tmp_path, {"domain": "", "mode": "local", "port": 8765}
        )
    save.assert_not_called()
    assert not any(
        "stop" in call.args[1] or "migrate" in call.args[1] for call in stages.call_args_list
    )


def test_migration_failure_does_not_start_services_or_retry(installer, tmp_path, monkeypatch):
    """Functionality: Preserve a failed migration instead of starting or retrying automatically.
    Inputs: Mocked preflight/backup/stop and failed migration. Outputs: Ordered boundary.
    Logic: Fail the seventh external stage; assert no later up/check call is attempted.
    Constraints: Does not execute real migrations or claim database rollback verification.
    """
    stages = Mock(side_effect=[None] * 6 + [subprocess.CalledProcessError(1, "migration")])
    monkeypatch.setattr(installer, "run", stages)
    monkeypatch.setattr(installer, "backup", Mock())
    with pytest.raises(subprocess.CalledProcessError):
        installer.deploy(
            ["docker", "compose"], tmp_path, {"domain": "", "mode": "local", "port": 8765}
        )
    assert stages.call_count == 7
    assert "stop" in stages.call_args_list[-2].args[1]
    assert "migrate" in stages.call_args_list[-1].args[1]


def test_backup_is_binary_and_failure_keeps_partial(installer, tmp_path):
    """Functionality: Preserve raw pg_dump data and mark failed output as incomplete.
    Inputs: A real Python child emitting binary data, followed by one exiting nonzero.
    Outputs: Exact-byte successful dump and a retained .partial on failure.
    Logic: Substitute only Docker argv with controlled local child processes.
    Constraints: Does not verify a real PostgreSQL restore; integration covers the actual database.
    """
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'PGDMP\\x00\\xff')",
        "--",
    ]
    path = installer.backup(command, tmp_path)
    assert path.read_bytes() == b"PGDMP\x00\xff"
    command = [sys.executable, "-c", "import sys; sys.exit(9)", "--"]
    with pytest.raises(subprocess.CalledProcessError):
        installer.backup(command, tmp_path)
    assert len(list((tmp_path / "backups").glob("*.partial"))) == 1


def test_literal_file_rejects_duplicates_and_keeps_values(installer, tmp_path):
    """Functionality: Keep secret metacharacters literal and reject ambiguous duplicate entries.
    Inputs: Synthetic UTF-8 config files. Outputs: Exact string or ValueError.
    Logic: Exercise the file parser directly. Constraints: No content is evaluated as shell code.
    """
    path = tmp_path / "model.env"
    path.write_text("KEY=$value#still=value\n", encoding="utf-8")
    assert installer.read_environment(path) == {"KEY": "$value#still=value"}
    path.write_text("KEY=first\nKEY=second\n", encoding="utf-8")
    with pytest.raises(ValueError):
        installer.read_environment(path)


def test_runtime_settings_preserve_server_security():
    """Functionality: Confirm local HTTP is explicit and both modes retain production storage/auth.
    Inputs: Isolated Python processes with synthetic environment. Outputs: Settings assertions.
    Logic: Import settings without a database connection; compare local/server/invalid modes.
    Constraints: Real TLS, PostgreSQL access and browser cookie behavior require integration checks.
    """
    env = os.environ | {
        "DJANGO_SECRET_KEY": "synthetic-only",
        "DJANGO_ALLOWED_HOSTS": "example.com",
        "POSTGRES_DB": "test",
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_HOST": "db",
        "POSTGRES_PORT": "5432",
        "CELERY_BROKER_URL": "redis://redis/0",
        "PDF_TASK_REDIS_URL": "redis://redis/1",
    }
    code = (
        "import json; from config import standalone as s; "
        "print(json.dumps([s.INTERVIEW_REQUIRE_LOGIN,s.DATABASES['default']['ENGINE'],"
        "s.SECURE_SSL_REDIRECT,s.SESSION_COOKIE_SECURE,s.CSRF_COOKIE_SECURE,"
        "s.PDF_TASK_EXECUTION]))"
    )
    for mode in ("local", "server"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT / "SystemCode/backend",
            env=env | {"DEPLOY_MODE": mode},
            capture_output=True,
            text=True,
            check=True,
        )
        values = json.loads(result.stdout)
        assert values == [
            True,
            "django.db.backends.postgresql",
            *([mode == "server"] * 3),
            "celery",
        ]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT / "SystemCode/backend",
        env=env | {"DEPLOY_MODE": "invalid"},
        capture_output=True,
    )
    assert result.returncode != 0


def test_parent_environment_cannot_publish_local_ports(installer, monkeypatch):
    """Functionality: Prevent ambient shell variables from changing saved network exposure.
    Inputs: Conflicting inherited Compose values and a mocked subprocess runner. Outputs: Clean env.
    Logic: Inspect the Docker environment, preserving its explicitly selected socket.
    Constraints: No actual port is opened; Compose integration tests verify generated configuration.
    """
    monkeypatch.setenv("HTTP_BIND", "0.0.0.0")
    monkeypatch.setenv("STATE_DIR", "/wrong")
    monkeypatch.setenv("COMPOSE_FILE", "/wrong/compose.yml")
    monkeypatch.setenv("DOCKER_HOST", "unix:///test.sock")
    runner = Mock()
    monkeypatch.setattr(installer.subprocess, "run", runner)
    installer.run("test", ["docker", "compose", "config", "--quiet"])
    env = runner.call_args.kwargs["env"]
    assert not {"HTTP_BIND", "STATE_DIR", "COMPOSE_FILE"}.intersection(env)
    assert env["DOCKER_HOST"] == "unix:///test.sock"


def test_server_hostname_is_normalized(installer, tmp_path):
    """Functionality: Match Django's lowercase Host comparison for mixed-case DNS input.
    Inputs: A mixed-case domain and temporary private configuration. Outputs: Lowercase saved host.
    Logic: Configure twice with equivalent DNS spelling. Constraints: No DNS or TLS request.
    """
    state = tmp_path / "state"
    state.mkdir()
    args = arguments(tmp_path, action="server", domain="Interview.EXAMPLE.com")
    metadata = installer.configure(args, state)
    assert metadata["domain"] == "interview.example.com"
    args.config = None
    args.domain = "INTERVIEW.example.COM"
    assert installer.configure(args, state) == metadata
