import copy
from pathlib import Path

import pytest

from dapi2marker.data.manifest import (
    build_manifest,
    read_manifest,
    select_samples,
    validate_manifest,
)


def test_three_folds_cover_rois_once_and_exclude_original_test(paired_sources, tmp_path):
    path = tmp_path / "manifest.json"
    first = build_manifest(paired_sources, path)
    assert first == build_manifest(paired_sources, path)
    manifest = read_manifest(path)
    all_val = []
    for fold in range(3):
        train, val = (
            select_samples(manifest, fold=fold),
            select_samples(manifest, fold=fold, split="val"),
        )
        assert {r["group_id"] for r in train}.isdisjoint(r["group_id"] for r in val)
        assert {r["tissue"] for r in val} == {"colon", "liver"}
        assert len(train) == 8 and len(val) == 4
        all_val.extend(r["sample_id"] for r in val)
    assert len(all_val) == len(set(all_val)) == 12
    assert len(select_samples(manifest, full=True)) == 12
    assert select_samples(manifest, full=True, split="val") == []
    test = select_samples(manifest, split="test")
    assert len(test) == 2 and all(r["target_path"] is None and r["fold"] is None for r in test)
    # Same bare ROI/tile name across tissues remains two independent samples.
    assert len([r for r in manifest["samples"] if r["roi"] == "ROI000" and r["col"] == 0]) == 2


def test_split_fingerprint_detects_membership_changes(paired_sources, tmp_path):
    manifest = build_manifest(paired_sources, tmp_path / "manifest.json")
    modified = copy.deepcopy(manifest)
    modified["samples"][0]["dapi_sha256"] = "altered"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_manifest(modified)


def test_missing_label_and_overlapping_test_roi_rejected(paired_sources, tmp_path):
    root = Path(paired_sources["colon"])
    label = root / "train" / "CD45RO" / "ROI001_00_00.png"
    contents = label.read_bytes()
    label.unlink()
    with pytest.raises(FileNotFoundError, match="label"):
        build_manifest(paired_sources, tmp_path / "missing.json")
    label.write_bytes(contents)
    (root / "test" / "DAPI" / "ROI099_00_00.png").rename(
        root / "test" / "DAPI" / "ROI001_01_00.png"
    )
    with pytest.raises(ValueError, match="both labeled and original test"):
        build_manifest(paired_sources, tmp_path / "overlap.json")
