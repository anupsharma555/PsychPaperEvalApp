from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_APP_PATH = PROJECT_ROOT / "scripts" / "run_app.py"


def _load_run_app_module():
    spec = importlib.util.spec_from_file_location("run_app", RUN_APP_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_app_defaults_backend_to_local_provider() -> None:
    run_app = _load_run_app_module()

    env = run_app._backend_env({"LLM_PROVIDER": "openai"})  # noqa: SLF001 - launch contract coverage

    assert env["LLM_PROVIDER"] == "local"
    assert env["PAPER_EVAL_ROOT"] == str(PROJECT_ROOT)
    assert env["PYTHONFAULTHANDLER"] == "1"


def test_run_app_launcher_lock_blocks_concurrent_start(monkeypatch, tmp_path) -> None:
    run_app = _load_run_app_module()
    monkeypatch.setattr(run_app, "ROOT", tmp_path)

    first = run_app._acquire_launcher_lock()  # noqa: SLF001
    assert first is not None
    try:
        assert run_app._acquire_launcher_lock() is None  # noqa: SLF001
    finally:
        run_app._release_launcher_lock(first)  # noqa: SLF001

    second = run_app._acquire_launcher_lock()  # noqa: SLF001
    assert second is not None
    run_app._release_launcher_lock(second)  # noqa: SLF001


def test_run_app_launcher_lock_is_scoped_by_backend_port(monkeypatch, tmp_path) -> None:
    run_app = _load_run_app_module()
    monkeypatch.setattr(run_app, "ROOT", tmp_path)

    port_8000 = run_app._acquire_launcher_lock(8000)  # noqa: SLF001
    assert port_8000 is not None
    try:
        port_19181 = run_app._acquire_launcher_lock(19181)  # noqa: SLF001
        assert port_19181 is not None
        run_app._release_launcher_lock(port_19181)  # noqa: SLF001
        assert run_app._pid_file(8000).name == "pids.8000.json"  # noqa: SLF001
        assert run_app._pid_file(19181).name == "pids.19181.json"  # noqa: SLF001
    finally:
        run_app._release_launcher_lock(port_8000)  # noqa: SLF001


def test_run_app_can_explicitly_launch_local_provider() -> None:
    run_app = _load_run_app_module()

    env = run_app._backend_env({"LLM_PROVIDER": "openai"}, provider="local")  # noqa: SLF001

    assert env["LLM_PROVIDER"] == "local"


def test_run_app_can_explicitly_launch_openai_provider() -> None:
    run_app = _load_run_app_module()

    env = run_app._backend_env({}, provider="openai")  # noqa: SLF001 - launch contract coverage

    assert env["LLM_PROVIDER"] == "openai"


def test_run_app_env_mode_preserves_environment_provider() -> None:
    run_app = _load_run_app_module()

    env = run_app._backend_env({"LLM_PROVIDER": "openai"}, provider="env")  # noqa: SLF001

    assert env["LLM_PROVIDER"] == "openai"


def test_run_app_env_mode_defaults_to_local_when_unset() -> None:
    run_app = _load_run_app_module()

    env = run_app._backend_env({}, provider="env")  # noqa: SLF001 - launch contract coverage

    assert env["LLM_PROVIDER"] == "local"


def test_api_only_launcher_does_not_start_frontend_or_browser_by_default() -> None:
    source = RUN_APP_PATH.read_text()
    web_start = source.index("if args.web:")
    web_block = source[web_start : source.index("except Exception as exc:", web_start)]
    browser_start = source.index("if args.open_browser:")
    browser_block = source[browser_start:]

    assert '"--api-only"' in source
    assert "if args.web:" in source
    assert "frontend_proc = subprocess.Popen" in web_block
    assert "if args.open_browser:" in source
    assert "subprocess.Popen([\"open\", frontend_url]" in browser_block


def test_run_app_does_not_claim_already_ready_grobid(monkeypatch) -> None:
    run_app = _load_run_app_module()
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(run_app, "_manage_grobid_enabled", lambda: True)
    monkeypatch.setattr(run_app, "_grobid_host_port", lambda: "8070")
    monkeypatch.setattr(run_app, "_grobid_health_url", lambda: "http://localhost:8070/api/isalive")
    monkeypatch.setattr(run_app, "_grobid_url", lambda: "http://localhost:8070")
    monkeypatch.setattr(run_app, "_http_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(run_app, "_docker_ready", lambda: True)
    monkeypatch.setattr(run_app, "_find_adoptable_grobid_container", lambda *_args, **_kwargs: "papereval-grobid")
    monkeypatch.setattr(run_app, "_log_lifecycle", lambda event, details=None: events.append((event, details or {})))

    assert run_app._start_grobid_container() is None  # noqa: SLF001
    assert events == [
        ("grobid_already_ready", {"container": "papereval-grobid", "url": "http://localhost:8070"})
    ]


def test_run_app_claims_new_grobid_container(monkeypatch) -> None:
    run_app = _load_run_app_module()
    readiness = iter([False, True])

    monkeypatch.setattr(run_app, "_manage_grobid_enabled", lambda: True)
    monkeypatch.setattr(run_app, "_grobid_host_port", lambda: "8070")
    monkeypatch.setattr(run_app, "_grobid_health_url", lambda: "http://localhost:8070/api/isalive")
    monkeypatch.setattr(run_app, "_grobid_url", lambda: "http://localhost:8070")
    monkeypatch.setattr(run_app, "_http_ready", lambda *_args, **_kwargs: next(readiness))
    monkeypatch.setattr(run_app, "_wait_for_docker", lambda: True)
    monkeypatch.setattr(run_app, "_find_adoptable_grobid_container", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_app, "_docker_args", lambda *args: ["docker", *args])
    monkeypatch.setattr(run_app, "_docker_container_exists", lambda *_args: False)
    monkeypatch.setattr(run_app.subprocess, "run", lambda *_args, **_kwargs: type("Result", (), {"returncode": 0})())
    monkeypatch.setattr(run_app, "_log_lifecycle", lambda *_args, **_kwargs: None)

    assert run_app._start_grobid_container() == "papereval-grobid"  # noqa: SLF001


def test_run_app_stops_grobid_by_default(monkeypatch) -> None:
    run_app = _load_run_app_module()
    events: list[tuple[str, dict]] = []
    calls: list[tuple] = []

    monkeypatch.delenv("PAPER_EVAL_STOP_GROBID_ON_EXIT", raising=False)
    monkeypatch.setattr(run_app, "_manage_grobid_enabled", lambda: True)
    monkeypatch.setattr(run_app, "_docker_args", lambda *args: ["docker", *args])
    monkeypatch.setattr(run_app.subprocess, "run", lambda *args, **_kwargs: calls.append(args))
    monkeypatch.setattr(run_app, "_log_lifecycle", lambda event, details=None: events.append((event, details or {})))

    run_app._stop_grobid_container("papereval-grobid")  # noqa: SLF001

    assert calls
    assert events == [("grobid_stopped", {"container": "papereval-grobid"})]


def test_run_app_can_leave_grobid_running_when_configured(monkeypatch) -> None:
    run_app = _load_run_app_module()
    events: list[tuple[str, dict]] = []

    monkeypatch.setenv("PAPER_EVAL_STOP_GROBID_ON_EXIT", "false")
    monkeypatch.setattr(run_app, "_manage_grobid_enabled", lambda: True)
    monkeypatch.setattr(run_app, "_docker_args", lambda *_args: (_ for _ in ()).throw(AssertionError("docker stop called")))
    monkeypatch.setattr(run_app, "_log_lifecycle", lambda event, details=None: events.append((event, details or {})))

    run_app._stop_grobid_container("papereval-grobid")  # noqa: SLF001

    assert events == [
        ("grobid_stop_skipped", {"container": "papereval-grobid", "reason": "persistent_by_default"})
    ]


def test_run_app_does_not_claim_existing_grobid_by_default(monkeypatch) -> None:
    run_app = _load_run_app_module()
    events: list[tuple[str, dict]] = []

    monkeypatch.delenv("PAPER_EVAL_STOP_GROBID_ON_EXIT", raising=False)
    monkeypatch.setattr(run_app, "_manage_grobid_enabled", lambda: True)
    monkeypatch.setattr(run_app, "_grobid_host_port", lambda: "8070")
    monkeypatch.setattr(run_app, "_grobid_health_url", lambda: "http://localhost:8070/api/isalive")
    monkeypatch.setattr(run_app, "_grobid_url", lambda: "http://localhost:8070")
    monkeypatch.setattr(run_app, "_http_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(run_app, "_docker_ready", lambda: True)
    monkeypatch.setattr(run_app, "_find_adoptable_grobid_container", lambda *_args, **_kwargs: "papereval-grobid")
    monkeypatch.setattr(run_app, "_log_lifecycle", lambda event, details=None: events.append((event, details or {})))

    assert run_app._start_grobid_container() is None  # noqa: SLF001
    assert events == [
        ("grobid_already_ready", {"container": "papereval-grobid", "url": "http://localhost:8070"})
    ]


def test_run_app_starts_docker_desktop_hidden_by_default(monkeypatch) -> None:
    run_app = _load_run_app_module()
    calls: list[list[str]] = []

    monkeypatch.setattr(run_app.sys, "platform", "darwin")
    monkeypatch.delenv("PAPER_EVAL_START_DOCKER_DESKTOP", raising=False)
    monkeypatch.delenv("PAPER_EVAL_DOCKER_DESKTOP_HIDDEN", raising=False)
    monkeypatch.setattr(
        run_app.subprocess,
        "Popen",
        lambda cmd, **_kwargs: calls.append(cmd),
    )

    run_app._start_docker_desktop()  # noqa: SLF001

    assert calls == [["open", "-gj", "-a", "Docker"]]
