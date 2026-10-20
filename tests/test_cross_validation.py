import json
from pathlib import Path

import yaml

from dapi2marker.cli import main


def test_cross_validation_aggregates_complete_oof_then_trains_full(experiment, tmp_path):
    experiment["train"]["max_steps"] = 1
    experiment["data"]["max_train_samples"] = None
    experiment["data"]["max_val_samples"] = None
    path = tmp_path / "cv.yaml"
    path.write_text(yaml.safe_dump(experiment))
    assert main(["cross-validate", "--config", str(path), "--include-full"]) == 0
    root = Path(experiment["paths"]["output_dir"])
    report = json.loads((root / "cross_validation.json").read_text())
    assert len(report["folds"]) == 3
    assert report["out_of_fold"]["tiles"] == 12
    assert len(report["out_of_fold"]["rois"]) == 6
    assert report["suggested_full_max_steps"] == 1
    full = json.loads((root / "full" / "result.json").read_text())
    assert full["train_tiles"] == 12 and full["val_tiles"] == 0
    assert (root / "full" / "model_final.pt").exists()


def test_invalid_full_budget_fails_before_any_training(experiment, tmp_path):
    path = tmp_path / "cv.yaml"
    path.write_text(yaml.safe_dump(experiment))
    assert (
        main(["cross-validate", "--config", str(path), "--include-full", "--full-max-steps", "0"])
        == 1
    )
    assert not Path(experiment["paths"]["output_dir"]).exists()
