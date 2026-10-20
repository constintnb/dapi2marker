"""Native-pixel crop lookup restricted to the selected training records."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image


def local_rng(seed, epoch, sample_id, stream):
    digest = hashlib.blake2b(
        f"{seed}:{epoch}:{sample_id}:{stream}".encode(), digest_size=8
    ).digest()
    return random.Random(int.from_bytes(digest, "little"))


class AdjacentCrop:
    def __init__(self, train_records, options, seed=2026):
        if options.get("layout") != "roi_row_col_stride256":
            raise ValueError("Adjacent crops require verified roi_row_col_stride256 layout.")
        self.size = 256
        self.tissues = set(options.get("tissues", ("liver",)))
        self.probability = options.get("probability", 0.5)
        self.max_shift = options.get("max_shift_pixels", 64)
        self.attempts = options.get("max_attempts", 16)
        self.seed = seed
        if (
            not 0 <= self.probability <= 1
            or not 1 <= self.max_shift < self.size
            or self.attempts < 1
        ):
            raise ValueError("Invalid adjacent crop probability, shift, or attempt count.")
        self.tiles = {}
        for record in train_records:
            if self.supports(record):
                if record["target_path"] is None or record["image_size"] != [256, 256]:
                    raise ValueError("Adjacent crops need labeled 256x256 records.")
                key = (record["tissue"], record["roi"], record["row"], record["col"])
                if key in self.tiles:
                    raise ValueError(f"Duplicate training spatial key: {key}")
                self.tiles[key] = record
        megabytes = options.get("cache_megabytes_per_worker", 32)
        if type(megabytes) not in (int, float) or megabytes < 0:
            raise ValueError("Crop cache size must be nonnegative.")
        self.cache_limit = int(megabytes * 1024**2)
        self.cache, self.cache_bytes = OrderedDict(), 0

    def supports(self, record):
        return record["tissue"] in self.tissues

    def plan(self, record, dx, dy):
        tissue, roi = record["tissue"], record["roi"]
        x, y = record["col"] * self.size + dx, record["row"] * self.size + dy
        result = []
        for row in range(y // self.size, (y + self.size - 1) // self.size + 1):
            for col in range(x // self.size, (x + self.size - 1) // self.size + 1):
                neighbor = self.tiles.get((tissue, roi, row, col))
                if neighbor is None:
                    return None
                x0, y0 = max(x, col * self.size), max(y, row * self.size)
                x1, y1 = (
                    min(x + self.size, (col + 1) * self.size),
                    min(y + self.size, (row + 1) * self.size),
                )
                source = (
                    slice(y0 - row * self.size, y1 - row * self.size),
                    slice(x0 - col * self.size, x1 - col * self.size),
                )
                dest = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
                result.append((neighbor, source, dest))
        return result

    def _read_pair(self, record):
        key = record["sample_id"]
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        pair = []
        for path_key in ("dapi_path", "target_path"):
            with Image.open(Path(record["source_root"]) / record[path_key]) as image:
                if image.size != (self.size, self.size):
                    raise ValueError(f"Source tile changed dimensions: {record['sample_id']}")
                pair.append(np.array(image.convert("L"), copy=True))
        pair = tuple(pair)
        size = sum(image.nbytes for image in pair)
        if size <= self.cache_limit:
            while self.cache and self.cache_bytes + size > self.cache_limit:
                _, evicted = self.cache.popitem(last=False)
                self.cache_bytes -= sum(image.nbytes for image in evicted)
            self.cache[key] = pair
            self.cache_bytes += size
        return pair

    def sample(self, record, epoch):
        rng = local_rng(self.seed, epoch, record["sample_id"], "crop")
        attempted = rng.random() < self.probability
        plan = None
        dx = dy = 0
        if attempted:
            for _ in range(self.attempts):
                dx, dy = (
                    rng.randint(-self.max_shift, self.max_shift),
                    rng.randint(-self.max_shift, self.max_shift),
                )
                if dx == dy == 0:
                    continue
                plan = self.plan(record, dx, dy)
                if plan is not None:
                    break
        if plan is None:
            dx = dy = 0
            plan = self.plan(record, 0, 0)
        if plan is None:
            raise ValueError(f"Anchor is outside training crop index: {record['sample_id']}")
        images = [np.empty((self.size, self.size), dtype=np.uint8) for _ in range(2)]
        for neighbor, source, dest in plan:
            for output, image in zip(images, self._read_pair(neighbor)):
                output[dest] = image[source]
        return tuple(images), {
            "attempted": attempted,
            "shifted": bool(dx or dy),
            "source_tiles": len(plan),
        }
