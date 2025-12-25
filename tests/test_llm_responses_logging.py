import os
from pathlib import Path


def test_tool_processor_import_has_no_log_side_effects(tmp_path, monkeypatch):
    """
    Importing ToolProcessor should not create empty `logs/llm_responses_*.log` files.

    We intentionally avoid import-time side effects to prevent confusing empty logs.
    """
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "logs").exists()

    # Import should not create logs directory or files.
    import midpoint.agents.tools.processor  # noqa: F401

    assert not (tmp_path / "logs").exists()


def test_llm_logger_writes_to_repo_root_logs(tmp_path, monkeypatch):
    """
    LLM responses logs should always go to `{repo_root}/logs` even if CWD is `.goal/`.
    """
    # Create a minimal "repo" root marker.
    (tmp_path / ".git").mkdir()
    (tmp_path / ".goal").mkdir()
    monkeypatch.chdir(tmp_path / ".goal")

    from midpoint.utils.llm_logging import configure_llm_responses_logger

    logger, log_path = configure_llm_responses_logger()
    logger.debug("hello from test")

    assert str(log_path).startswith(str(tmp_path / "logs"))
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8").strip() != ""

