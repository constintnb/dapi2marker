"""Paired multi-source records with deterministic paired augmentation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .geometry import AdjacentCrop, local_rng


def limit_by_tissue(records: list[dict], limit: int | None) -> list[dict]:
    if limit is None or limit >= len(records):
        return records
    tissues = defaultdict(list)
    for record in records:
        tissues[record["tissue"]].append(record)
    if limit < len(tissues):
        raise ValueError("A sample limit must allow at least one sample from every tissue.")
    selected, offset = [], 0
    while len(selected) < limit:
        for tissue in sorted(tissues):
            if offset < len(tissues[tissue]) and len(selected) < limit:
                selected.append(tissues[tissue][offset])
        offset += 1
    return selected


class PairedDataset(Dataset):
    def __init__(self, records, *, augment=False, seed=2026, adjacent_crop=None, limit=None):
        self.records = limit_by_tissue(records, limit)
        if not self.records:
            raise ValueError("Dataset has no selected samples.")
        self.augment = augment
        self.seed = seed
        # Loaders recreate workers each epoch (persistent_workers=False), so the
        # ordinary epoch value is copied when workers start. A shared tensor
        # would unnecessarily launch torch_shm_manager even with zero workers.
        self._epoch = 0
        crop = adjacent_crop or {}
        self.cropper = (
            AdjacentCrop(self.records, crop, seed) if augment and crop.get("enabled") else None
        )
        for record in self.records:
            for key in ("dapi_path", "target_path"):
                if (
                    record[key] is not None
                    and not (Path(record["source_root"]) / record[key]).is_file()
                ):
                    raise FileNotFoundError(Path(record["source_root"]) / record[key])

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    @staticmethod
    def _read(record, key):
        with Image.open(Path(record["source_root"]) / record[key]) as image:
            return np.array(image.convert("L"), copy=True)

    def __getitem__(self, index):
        record = self.records[index]
        epoch = self._epoch
        meta = {"attempted": False, "shifted": False, "source_tiles": 1}
        if self.cropper is not None and self.cropper.supports(record):
            (dapi, target), meta = self.cropper.sample(record, epoch)
        else:
            dapi = self._read(record, "dapi_path")
            target = (
                self._read(record, "target_path") if record["target_path"] is not None else None
            )
        if target is not None and dapi.shape != target.shape:
            raise ValueError(f"Input/label shape mismatch: {record['sample_id']}")
        tensors = [torch.from_numpy(dapi.copy()).unsqueeze(0).float() / 127.5 - 1]
        if target is not None:
            tensors.append(torch.from_numpy(target.copy()).unsqueeze(0).float() / 127.5 - 1)
        if self.augment:
            rng = local_rng(self.seed, epoch, record["sample_id"], "geometry")
            if rng.random() > 0.5:
                tensors = [tensor.flip(-1) for tensor in tensors]
            if rng.random() > 0.5:
                tensors = [tensor.flip(-2) for tensor in tensors]
            rotations = rng.randrange(4)
            tensors = [torch.rot90(tensor, rotations, (-2, -1)) for tensor in tensors]
        result = {
            "dapi": tensors[0],
            "sample_index": index,
            "sample_id": record["sample_id"],
            "tissue": record["tissue"],
            "group_id": record["group_id"],
            "crop_metadata": meta,
        }
        if target is not None:
            result["targets"] = tensors[1]
        return result


def make_sampler(dataset, mode: str, generator: torch.Generator):
    if mode == "natural":
        return None
    if mode != "tissue_balanced":
        raise ValueError("data.sampling must be natural or tissue_balanced.")
    counts = Counter(record["tissue"] for record in dataset.records)
    weights = [1.0 / counts[record["tissue"]] for record in dataset.records]
    return WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
