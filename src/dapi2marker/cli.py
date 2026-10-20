"""Installed command line; no working-directory dependent script imports."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from . import __version__
from .config import load_config, run_directory, validate_config
from .data.manifest import build_manifest, read_manifest, summarize_manifest
from .models.factory import parameter_count


def _print(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _config_options(parser):
    parser.add_argument("--config", required=True, help="Path to a YAML preset")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))


def _budget_options(parser):
    parser.add_argument("--max-steps", type=int, help="Override the fixed optimizer-step budget")


def parser():
    root = argparse.ArgumentParser(prog="dapi2marker", description=__doc__)
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in (
        "params",
        "plan",
        "prepare-splits",
        "train",
        "evaluate",
        "predict",
        "cross-validate",
        "download-vae",
        "doctor",
    ):
        command = commands.add_parser(name)
        if name == "doctor":
            command.add_argument(
                "--require-cuda",
                action="store_true",
                help="Fail unless visible CUDA GPUs pass a tiny forward/backward check",
            )
            continue
        _config_options(command)
        if name == "prepare-splits":
            command.add_argument(
                "--force", action="store_true", help="Explicitly replace changed membership"
            )
        elif name == "train":
            modes = command.add_mutually_exclusive_group()
            modes.add_argument("--fold", type=int, default=0)
            modes.add_argument("--full", action="store_true")
            command.add_argument(
                "--resume", type=Path, help="Trusted local training-state checkpoint"
            )
            _budget_options(command)
        elif name in {"evaluate", "predict"}:
            command.add_argument("--checkpoint", required=True, type=Path)
            command.add_argument("--fold", type=int, default=0)
            command.add_argument(
                "--split", choices=("val", "test"), default="val" if name == "evaluate" else "test"
            )
            command.add_argument("--output", type=Path)
        elif name == "cross-validate":
            command.add_argument(
                "--include-full",
                action="store_true",
                help="Run a fresh full-data model after all folds",
            )
            command.add_argument(
                "--full-max-steps", type=int, help="Full-data budget, otherwise the same as CV"
            )
            command.add_argument("--dry-run", action="store_true")
            _budget_options(command)
    return root


def cross_validate(config, args):
    if args.full_max_steps is not None:
        if not args.include_full or args.full_max_steps < 1:
            raise ValueError("A positive --full-max-steps requires --include-full.")
    manifest = read_manifest(config["paths"]["manifest"], sources=config["data"]["sources"])
    steps = args.max_steps or config["train"]["max_steps"]
    commands = []
    for fold in range(manifest["n_folds"]):
        commands.append(
            [
                sys.executable,
                "-m",
                "dapi2marker",
                "train",
                "--config",
                args.config,
                "--fold",
                str(fold),
                "--max-steps",
                str(steps),
            ]
        )
    if args.include_full:
        commands.append(
            [
                sys.executable,
                "-m",
                "dapi2marker",
                "train",
                "--config",
                args.config,
                "--full",
                "--max-steps",
                str(args.full_max_steps or steps),
            ]
        )
    if args.device:
        for command in commands:
            command.extend(["--device", args.device])
    if args.dry_run:
        _print(
            {
                "data": summarize_manifest(manifest),
                "commands": commands,
                "full_budget_policy": "explicit --full-max-steps or same optimizer-step budget as each fold",
            }
        )
        return
    if args.full_max_steps is not None and not args.include_full:
        raise ValueError("--full-max-steps requires --include-full.")
    # Fail before starting any training if a target run is already occupied.
    for fold in range(manifest["n_folds"]):
        directory = run_directory(config, fold)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"CV run exists: {directory}. Resume folds individually or choose a new output_dir."
            )
    full_directory = run_directory(config, full=True)
    if args.include_full and full_directory.exists() and any(full_directory.iterdir()):
        raise FileExistsError(f"Full run exists: {full_directory}.")
    for command in commands[: manifest["n_folds"]]:
        subprocess.run(command, check=True)
    from .metrics import METRICS, aggregate_metrics, read_rows, write_rows
    from .training import write_json
    import numpy as np

    reports, rows, best_steps = [], [], []
    for fold in range(manifest["n_folds"]):
        directory = run_directory(config, fold)
        reports.append(json.loads((directory / "validation_best.json").read_text()))
        rows.extend(read_rows(directory / "validation_best.csv"))
        best_steps.append(reports[-1]["step"])
    if config["data"].get("max_val_samples") is None:
        expected = {
            record["sample_id"]
            for record in manifest["samples"]
            if record["source_split"] != "test"
        }
        if expected != {row["sample_id"] for row in rows}:
            raise ValueError("OOF predictions do not cover the complete labeled pool exactly once.")
    report = {
        "folds": reports,
        "fold_macro_mean": {
            key: float(np.mean([r["macro"][key] for r in reports])) for key in METRICS
        },
        "fold_macro_std": {
            key: float(np.std([r["macro"][key] for r in reports], ddof=1)) for key in METRICS
        },
        "out_of_fold": aggregate_metrics(rows),
        "best_steps": best_steps,
        "suggested_full_max_steps": int(np.median(best_steps)),
        "full_max_steps": (args.full_max_steps or steps) if args.include_full else None,
        "manifest_fingerprint": manifest["fingerprint"],
    }
    output = Path(config["paths"]["output_dir"])
    write_json(report, output / "cross_validation.json")
    write_rows(rows, output / "out_of_fold.csv")
    _print(report)
    if args.include_full:
        subprocess.run(commands[-1], check=True)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from .diagnostics import runtime_report

            report = runtime_report()
            _print(report)
            return 1 if args.require_cuda and not report["cuda_usable"] else 0
        config = load_config(args.config)
        if args.device is not None:
            config["runtime"]["device"] = args.device
        if getattr(args, "max_steps", None) is not None:
            config["train"]["max_steps"] = args.max_steps
        validate_config(config)
        if args.command == "params":
            _print(
                {
                    "trainable_parameters": parameter_count(config["model"]),
                    "scope": "UNet + condition encoder; frozen VAE excluded",
                }
            )
        elif args.command == "prepare-splits":
            manifest = build_manifest(
                config["data"]["sources"],
                config["paths"]["manifest"],
                target=config["data"]["target"],
                n_folds=config["data"]["folds"],
                seed=config["data"]["seed"],
                overwrite=args.force,
            )
            _print(summarize_manifest(manifest))
        elif args.command == "plan":
            _print(
                summarize_manifest(
                    read_manifest(config["paths"]["manifest"], sources=config["data"]["sources"])
                )
            )
        elif args.command == "train":
            from .training import train_run

            _print(train_run(config, fold=args.fold, full=args.full, resume=args.resume))
        elif args.command in {"evaluate", "predict"}:
            from .evaluation import predict_run

            _print(
                predict_run(
                    config,
                    args.checkpoint,
                    fold=args.fold,
                    split=args.split,
                    output=args.output,
                    save_images=args.command == "predict",
                )
            )
        elif args.command == "cross-validate":
            cross_validate(config, args)
        elif args.command == "download-vae":
            from huggingface_hub import HfApi, snapshot_download

            repository = config["vae"]["repository"]
            revision = (
                HfApi().model_info(repository, revision=config["vae"].get("revision", "main")).sha
            )

            path = snapshot_download(
                repo_id=repository,
                revision=revision,
                local_dir=config["vae"]["path"],
                allow_patterns=[
                    "config.json",
                    "diffusion_pytorch_model.safetensors",
                    "README.md",
                    "LICENSE*",
                ],
            )
            provenance = {"vae_path": path, "repository": repository, "revision": revision}
            (Path(path) / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
            _print(provenance)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
