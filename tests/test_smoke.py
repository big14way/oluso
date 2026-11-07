import json

import pytest

from oluso import config, smoke
from oluso.cli import main
from oluso.hub import init_collections


@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "ROOT", tmp_path)


def test_mock_smoke_runs_without_credentials(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    res = smoke.run(backend="mock")
    assert res["yoruba_outputs"] == "3/3"
    assert res["gate_passed"] is False  # only the real target model can pass the gate
    written = json.loads((config.RUNS_DIR / "p0_smoke_mock.json").read_text())
    assert written["backend"] == "mock"


def test_cli_mock_exit_zero(capsys):
    assert main(["smoke", "--backend", "mock"]) == 0
    assert "gate_passed" in capsys.readouterr().out


def test_hub_init_is_dry_run_by_default():
    res = init_collections(apply=False)
    assert res["dry_run"] is True
    assert {v["title"] for v in res["plan"].values()} == {"oluso-models", "oluso-datasets"}
    assert all(v["private"] is False for v in res["plan"].values())


@pytest.mark.network
def test_tiny_model_exercises_hf_path():
    try:
        res = smoke.run(backend="tiny", max_new_tokens=8, device="cpu")
    except OSError as exc:  # offline
        pytest.skip(f"tiny test model unavailable: {exc}")
    assert len(res["rows"]) == 3
    assert all(r["new_tokens"] > 0 for r in res["rows"])
    assert res["gate_passed"] is False
