import copy
import json
from pathlib import Path

import pytest
import torch

from dapi2marker.checkpoint import load_weights
from dapi2marker.evaluation import predict_run
from dapi2marker.training import train_run


def test_train_evaluate_predict_and_full_data(experiment):
    result = train_run(experiment, fold=0)
    assert result["step"] == 2 and result["best_step"] in (1, 2)
    directory = Path(experiment["paths"]["output_dir"]) / "fold_0"
    report = json.loads((directory / "validation_best.json").read_text())
    assert report["tiles"] == 2 and set(report["tissues"]) == {"colon", "liver"}
    prediction = predict_run(experiment, directory / "model_best.pt", split="test")
    assert prediction["predicted_tiles"] == 2
    assert (directory / "predictions_test" / "colon" / "ROI099_00_00.png").is_file()
    assert (directory / "predictions_test" / "liver" / "ROI099_00_00.png").is_file()
    with pytest.raises(ValueError, match="this fold"):
        predict_run(experiment, directory / "model_best.pt", fold=1, save_images=False)
    full = train_run(experiment, full=True)
    assert full["val_tiles"] == 0 and full["best_step"] is None
    assert (directory.parent / "full" / "model_final.pt").is_file()
    assert not (directory.parent / "full" / "validation_best.json").exists()


def test_interrupted_resume_matches_continuous_training(experiment, monkeypatch):
    experiment["train"]["mixed_precision"] = "auto"
    reference = copy.deepcopy(experiment)
    reference["paths"]["output_dir"] += "_reference"
    train_run(reference, fold=0)
    import dapi2marker.training as training

    real_save = training.atomic_save

    def stop_after_first_checkpoint(value, path):
        real_save(value, path)
        if value.get("step") == 1:
            raise InterruptedError("simulated interruption after persisted training state")

    monkeypatch.setattr(training, "atomic_save", stop_after_first_checkpoint)
    with pytest.raises(InterruptedError):
        train_run(experiment, fold=0)
    monkeypatch.setattr(training, "atomic_save", real_save)
    directory = Path(experiment["paths"]["output_dir"]) / "fold_0"
    result = train_run(experiment, fold=0, resume=directory / "checkpoint_last.pt")
    assert result["step"] == 2
    restored, _ = load_weights(directory / "model_last.pt")
    expected, _ = load_weights(Path(reference["paths"]["output_dir"]) / "fold_0" / "model_last.pt")
    for key, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected.state_dict()[key], rtol=0, atol=0)
    changed = copy.deepcopy(experiment)
    changed["train"]["max_steps"] += 1
    with pytest.raises(ValueError, match="Resume requires identical"):
        train_run(changed, fold=0, resume=directory / "checkpoint_last.pt")
