from pathlib import Path
import subprocess
import sys

import pytest

from dapi2marker.cli import main
from dapi2marker.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_preset_paths_resolved_independently_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(ROOT / "configs" / "small.yaml")
    assert config["data"]["sources"]["colon"] == str(ROOT / "ds")
    assert config["paths"]["manifest"] == str(ROOT / "splits" / "joint_cd45ro.json")
    assert config["vae"]["path"] == str(ROOT / "weights" / "sd-vae-ft-mse")
    assert load_config(ROOT / "configs" / "baseline.yaml")["model"]["base_channels"] == 256
    assert load_config(ROOT / "configs" / "smoke.yaml")["train"]["mixed_precision"] == "no"


def test_cli_available_outside_repository(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dapi2marker",
            "params",
            "--config",
            str(ROOT / "configs" / "small.yaml"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "23560572" in result.stdout
    bad = subprocess.run(
        [sys.executable, "-m", "dapi2marker", "params", "--config", str(tmp_path / "missing.yaml")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1


def test_config_inheritance_cycle_rejected(tmp_path):
    (tmp_path / "one.yaml").write_text("extends: two.yaml\n")
    (tmp_path / "two.yaml").write_text("extends: one.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "one.yaml")


def test_dry_run_uses_three_fresh_fold_models_and_full(experiment, tmp_path, capsys):
    import yaml

    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment))
    assert main(["cross-validate", "--config", str(path), "--dry-run", "--include-full"]) == 0
    text = capsys.readouterr().out
    assert '"--full"' in text and text.count('"--fold"') == 3
    assert not Path(experiment["paths"]["output_dir"]).exists()
