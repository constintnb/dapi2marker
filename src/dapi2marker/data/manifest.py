"""Portable, fixed tissue-aware ROI folds over the original train/val pools."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TILE_NAME = re.compile(r"^(ROI\d+)_(\d+)_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def tile_key(filename: str) -> tuple[str, int, int]:
    match = TILE_NAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"Expected ROI_row_col image filename: {filename}")
    return match[1].upper(), int(match[2]), int(match[3])


def stable_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fingerprint(manifest: dict) -> str:
    return stable_digest(
        {key: manifest[key] for key in ("schema_version", "target", "seed", "n_folds", "samples")}
    )


def _assign_folds(samples: list[dict], n_folds: int, seed: int) -> dict[str, int]:
    sizes = Counter(record["group_id"] for record in samples if record["source_split"] != "test")
    tissues = sorted({record["tissue"] for record in samples})
    total_sizes, total_groups = [0] * n_folds, [0] * n_folds
    assignments = {}
    for tissue in tissues:
        groups = [group for group in sizes if group.startswith(tissue + ":")]
        if len(groups) < n_folds:
            raise ValueError(
                f"{tissue} needs at least {n_folds} labeled ROIs; found {len(groups)}."
            )
        quotient, remainder = divmod(len(groups), n_folds)
        extras = sorted(range(n_folds), key=lambda f: (total_groups[f], total_sizes[f], f))[
            :remainder
        ]
        capacity = [quotient + (fold in extras) for fold in range(n_folds)]
        tissue_sizes, tissue_groups = [0] * n_folds, [0] * n_folds
        # Largest groups first; the digest makes equal-size ties deterministic.
        groups.sort(key=lambda group: (-sizes[group], stable_digest([seed, group])))
        for group in groups:
            eligible = [f for f in range(n_folds) if tissue_groups[f] < capacity[f]]
            fold = min(eligible, key=lambda f: (tissue_sizes[f], total_sizes[f], f))
            assignments[group] = fold
            tissue_groups[fold] += 1
            tissue_sizes[fold] += sizes[group]
            total_groups[fold] += 1
            total_sizes[fold] += sizes[group]
    return assignments


def build_manifest(
    sources: dict[str, str],
    output: str | Path,
    *,
    target="CD45RO",
    n_folds=3,
    seed=2026,
    overwrite=False,
) -> dict:
    from PIL import Image

    if type(n_folds) is not int or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not sources or any(not re.fullmatch(r"[a-z][a-z0-9_-]*", key) for key in sources):
        raise ValueError("Source IDs must be nonempty lowercase names.")
    if not target or Path(target).name != target or target in {".", ".."}:
        raise ValueError("target must be a plain channel directory name.")
    output = Path(output).expanduser().resolve()
    roots = {key: Path(value).expanduser().resolve() for key, value in sources.items()}
    samples, identities = [], set()
    for tissue, root in sorted(roots.items()):
        for split in ("train", "val", "test"):
            directory = root / split / "DAPI"
            if not directory.is_dir():
                if split == "test":
                    continue
                raise FileNotFoundError(f"DAPI directory is missing: {directory}")
            images = sorted(
                p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            for image in images:
                roi, row, col = tile_key(image.name)
                sample_id = f"{tissue}:{roi}:{row}:{col}:{target}"
                if sample_id in identities:
                    raise ValueError(f"Duplicate spatial sample across source splits: {sample_id}")
                identities.add(sample_id)
                label = root / split / target / image.name if split != "test" else None
                if label is not None and not label.is_file():
                    raise FileNotFoundError(f"Paired {target} label is missing: {label}")
                with Image.open(image) as handle:
                    size = handle.size
                    handle.verify()
                if min(size) < 32 or any(value % 8 for value in size):
                    raise ValueError(f"Image dimensions must be multiples of 8 and >=32: {image}")
                if label is not None:
                    with Image.open(label) as handle:
                        if handle.size != size:
                            raise ValueError(f"Input/label dimension mismatch: {image}")
                        handle.verify()
                samples.append(
                    {
                        "sample_id": sample_id,
                        "group_id": f"{tissue}:{roi}",
                        "tissue": tissue,
                        "roi": roi,
                        "row": row,
                        "col": col,
                        "source_split": split,
                        "filename": image.name,
                        "dapi_path": image.relative_to(root).as_posix(),
                        "target_path": label.relative_to(root).as_posix()
                        if label is not None
                        else None,
                        "image_size": list(size),
                        "dapi_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "target_sha256": hashlib.sha256(label.read_bytes()).hexdigest()
                        if label
                        else None,
                    }
                )
    if not samples:
        raise ValueError("No image samples were found.")
    for tissue in roots:
        labeled_groups = {
            record["group_id"]
            for record in samples
            if record["tissue"] == tissue and record["source_split"] != "test"
        }
        if len(labeled_groups) < n_folds:
            raise ValueError(
                f"{tissue} needs at least {n_folds} labeled ROIs; found {len(labeled_groups)}."
            )
    assignments = _assign_folds(samples, n_folds, seed)
    for record in samples:
        record["fold"] = (
            assignments.get(record["group_id"]) if record["source_split"] != "test" else None
        )
    result = {
        "schema_version": 1,
        "target": target,
        "n_folds": n_folds,
        "seed": seed,
        "method": "tissue-stratified complete-ROI folds; size balancing with seeded tie breaks",
        "sources": {key: os.path.relpath(root, output.parent) for key, root in roots.items()},
        "samples": samples,
    }
    result["fingerprint"] = _fingerprint(result)
    validate_manifest(result)
    if output.exists() and not overwrite:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != result:
            raise FileExistsError(
                f"Manifest already exists with different membership: {output}; use --force."
            )
        return result
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("n_folds", 0) < 2:
        raise ValueError("Unsupported manifest schema or fold count.")
    identities, group_folds, group_kinds = set(), {}, defaultdict(set)
    tissue_folds = defaultdict(set)
    for record in manifest["samples"]:
        sid = record["sample_id"]
        if sid in identities:
            raise ValueError(f"Duplicate sample ID: {sid}")
        identities.add(sid)
        tissue, roi, row, col = record["tissue"], record["roi"], record["row"], record["col"]
        if tissue not in manifest["sources"]:
            raise ValueError(f"Unknown source: {tissue}")
        if (
            record["group_id"] != f"{tissue}:{roi}"
            or sid != f"{tissue}:{roi}:{row}:{col}:{manifest['target']}"
        ):
            raise ValueError(f"Inconsistent sample identity: {sid}")
        if tile_key(record["filename"]) != (roi, row, col):
            raise ValueError(f"Inconsistent filename coordinates: {sid}")
        for key in ("dapi_path", "target_path"):
            value = record[key]
            if value is not None and (Path(value).is_absolute() or ".." in Path(value).parts):
                raise ValueError(f"Sample paths must be relative within the source: {sid}")
        group = record["group_id"]
        if record["source_split"] == "test":
            if record["target_path"] is not None or record["fold"] is not None:
                raise ValueError("Original test samples must remain unlabeled and outside CV.")
            group_kinds[group].add("test")
        elif record["source_split"] in {"train", "val"}:
            fold = record["fold"]
            if (
                type(fold) is not int
                or not 0 <= fold < manifest["n_folds"]
                or record["target_path"] is None
            ):
                raise ValueError(f"Invalid labeled fold record: {sid}")
            if group in group_folds and group_folds[group] != fold:
                raise ValueError(f"ROI split across folds: {group}")
            group_folds[group] = fold
            tissue_folds[tissue].add(fold)
            group_kinds[group].add("labeled")
        else:
            raise ValueError(f"Unknown original source split: {sid}")
    if any(len(kinds) > 1 for kinds in group_kinds.values()):
        raise ValueError("A source ROI occurs in both labeled and original test pools.")
    if set(tissue_folds) != set(manifest["sources"]) or any(
        folds != set(range(manifest["n_folds"])) for folds in tissue_folds.values()
    ):
        raise ValueError("Every tissue must have labeled ROI groups in every fold.")
    if manifest.get("fingerprint") != _fingerprint(manifest):
        raise ValueError("Manifest fingerprint does not match its membership.")


def read_manifest(path: str | Path, *, sources: dict[str, str] | None = None) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run dapi2marker prepare-splits first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    roots = sources if sources is not None else manifest["sources"]
    manifest["resolved_sources"] = {
        key: str(Path(value) if Path(value).is_absolute() else (path.parent / value).resolve())
        for key, value in roots.items()
    }
    if set(manifest["sources"]) != set(manifest["resolved_sources"]):
        raise ValueError("Configured source IDs do not match the manifest.")
    return manifest


def select_samples(manifest: dict, *, split="train", fold=0, full=False) -> list[dict]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test.")
    if (
        not full
        and split != "test"
        and (type(fold) is not int or not 0 <= fold < manifest["n_folds"])
    ):
        raise ValueError("Selected fold is outside the manifest fold range.")
    selected = []
    for record in manifest["samples"]:
        test = record["source_split"] == "test"
        include = (
            test
            if split == "test"
            else (
                not test
                and (split == "train" if full else (record["fold"] == fold) == (split == "val"))
            )
        )
        if include:
            selected.append(
                {**record, "source_root": manifest["resolved_sources"][record["tissue"]]}
            )
    return selected


def summarize_manifest(manifest: dict) -> dict:
    result = {"target": manifest["target"], "fingerprint": manifest["fingerprint"], "folds": []}
    for fold in range(manifest["n_folds"]):
        counts = Counter()
        groups = defaultdict(set)
        for record in manifest["samples"]:
            if record["source_split"] == "test":
                continue
            split = "val" if record["fold"] == fold else "train"
            counts[(split, record["tissue"])] += 1
            groups[(split, record["tissue"])].add(record["group_id"])
        result["folds"].append(
            {
                "fold": fold,
                "train": {t: counts[("train", t)] for t in manifest["sources"]},
                "val": {t: counts[("val", t)] for t in manifest["sources"]},
                "val_groups": {t: sorted(groups[("val", t)]) for t in manifest["sources"]},
            }
        )
    result["full_train"] = dict(
        Counter(r["tissue"] for r in manifest["samples"] if r["source_split"] != "test")
    )
    result["test"] = dict(
        Counter(r["tissue"] for r in manifest["samples"] if r["source_split"] == "test")
    )
    return result
