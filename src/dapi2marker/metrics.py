"""Image metrics and ROI/tissue-macro aggregation for model selection."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .losses import s50_score


METRICS = ("psnr", "ssim", "s50")


def image_metrics(target, prediction):
    psnr = float(peak_signal_noise_ratio(target, prediction, data_range=255))
    ssim = float(structural_similarity(target, prediction, data_range=255))
    # Perfect reconstructions have infinite PSNR. Store a finite capped value;
    # this agrees with the 50 dB cap used by S50 and keeps JSON portable.
    return {"psnr": min(psnr, 100.0), "ssim": ssim, "s50": float(s50_score(psnr, ssim))}


def _average(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def aggregate_metrics(rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty validation set.")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate evaluation sample IDs.")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["tissue"], row["group_id"])].append(row)
    rois = [
        {"tissue": tissue, "group_id": group, "tiles": len(items), **_average(items)}
        for (tissue, group), items in sorted(groups.items())
    ]
    tissues = {
        tissue: {
            "tiles": sum(roi["tiles"] for roi in rois if roi["tissue"] == tissue),
            "rois": sum(roi["tissue"] == tissue for roi in rois),
            **_average([roi for roi in rois if roi["tissue"] == tissue]),
        }
        for tissue in sorted({roi["tissue"] for roi in rois})
    }
    return {
        "tiles": len(rows),
        "rois": rois,
        "tissues": tissues,
        "macro": _average(list(tissues.values())),
        "micro": _average(rows),
        "selection_metric": "macro.s50",
        "weighting": "equal tissue, equal ROI within tissue",
    }


def write_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "tissue", "group_id", *METRICS))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            {**row, **{key: float(row[key]) for key in METRICS}} for row in csv.DictReader(handle)
        ]
