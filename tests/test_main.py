"""The CLI entry point, exercised offline via --fake-llm."""

from __future__ import annotations

import json

import main
from simulation.runlog import (
    CONFIG_FILENAME,
    MESSAGES_FILENAME,
    ROUNDS_FILENAME,
    SUMMARY_FILENAME,
    read_jsonl,
)


def run_cli(tmp_path, *extra):
    return main.main(
        ["--fake-llm", "--output-dir", str(tmp_path), "--log-level", "ERROR", *extra]
    )


def test_fake_run_succeeds_without_a_key_or_network(tmp_path):
    assert run_cli(tmp_path, "--rounds", "2") == 0

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    for filename in (
        CONFIG_FILENAME,
        MESSAGES_FILENAME,
        ROUNDS_FILENAME,
        SUMMARY_FILENAME,
    ):
        assert (run_dirs[0] / filename).is_file()


def test_fake_run_writes_the_default_roster(tmp_path):
    run_cli(tmp_path, "--rounds", "1")

    run_dir = next(iter(tmp_path.iterdir()))
    rows = read_jsonl(run_dir / MESSAGES_FILENAME)
    assert [row["sender"] for row in rows] == [
        "Government",
        "Industry",
        "NGO",
        "Scientist",
    ]


def test_cli_overrides_reach_the_resolved_config(tmp_path):
    run_cli(tmp_path, "--rounds", "1", "--sensitivity", "0.4", "--protocol", "sequential")

    run_dir = next(iter(tmp_path.iterdir()))
    saved = json.loads((run_dir / CONFIG_FILENAME).read_text("utf-8"))
    assert saved["rounds"] == 1
    assert saved["sensitivity"] == 0.4
    assert saved["protocol"] == "sequential"


def test_demo_run_exercises_retry_and_fallback_paths(tmp_path):
    """The offline demo deliberately injects faults, so artifacts show them."""
    run_cli(tmp_path, "--rounds", "5")

    run_dir = next(iter(tmp_path.iterdir()))
    summary = json.loads((run_dir / SUMMARY_FILENAME).read_text("utf-8"))

    assert summary["total_retries"] > 0
    assert summary["fallback_count"] > 0
    assert summary["attempt_status_counts"]["invalid_json"] > 0
    assert summary["attempt_status_counts"]["api_error"] > 0
    assert summary["attempt_status_counts"]["ok"] > 0
    # More calls than turns, because failed attempts were retried.
    assert summary["total_llm_calls"] > summary["total_agent_turns"]


def test_offline_run_is_identifiable_in_the_artifacts(tmp_path):
    """resolved_config keeps the configured model; summary names the real client."""
    run_cli(tmp_path, "--rounds", "1")

    run_dir = next(iter(tmp_path.iterdir()))
    saved = json.loads((run_dir / CONFIG_FILENAME).read_text("utf-8"))
    summary = json.loads((run_dir / SUMMARY_FILENAME).read_text("utf-8"))

    assert saved["model"] == "gpt-4o-mini"
    assert summary["model"] == "gpt-4o-mini"
    assert summary["llm_client"] == "FakeLLM"
    assert summary["llm_model"] == "fake-llm-demo"


def test_invalid_cli_configuration_exits_nonzero(tmp_path):
    assert run_cli(tmp_path, "--rounds", "0") == 2


def test_missing_config_file_exits_nonzero(tmp_path):
    assert run_cli(tmp_path, "--config", str(tmp_path / "nope.json")) == 2


def test_total_failure_returns_nonzero(tmp_path, monkeypatch):
    from utils.fake_llm import FakeApiError, FakeLLM

    monkeypatch.setattr(
        main, "make_demo_llm", lambda: FakeLLM([FakeApiError(retryable=True)], loop=True)
    )

    assert run_cli(tmp_path, "--rounds", "1") == 1


def test_report_warns_about_fallbacks(tmp_path, capsys):
    run_cli(tmp_path, "--rounds", "5")

    out = capsys.readouterr().out
    assert "===== RESULTS =====" in out
    assert "Average policy:" in out
    assert "Agreement score:" in out
    assert "WARNING:" in out
    assert "fell back after exhausting retries" in out


def test_report_shows_clean_runs_as_clean(tmp_path, capsys, monkeypatch):
    from utils.fake_llm import make_demo_llm

    monkeypatch.setattr(
        main,
        "make_demo_llm",
        lambda: make_demo_llm(malformed_calls=(), api_error_calls=()),
    )
    run_cli(tmp_path, "--rounds", "2")

    out = capsys.readouterr().out
    assert "Fallbacks: 0" in out
    assert "WARNING:" not in out


def test_fake_llm_flag_never_constructs_a_real_client(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("OpenAIClient must not be constructed with --fake-llm")

    monkeypatch.setattr(main, "OpenAIClient", explode)
    assert run_cli(tmp_path, "--rounds", "1") == 0
