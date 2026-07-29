#!/usr/bin/env python3
"""Verify trainer/rollout parity for a metadata QuanTrick checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import random_split

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from snn import train_v2
from student.bootstrap_backend import load_bootstrap_policy_from_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_hook_indices(model: torch.nn.Module) -> list[int]:
    weighted_blocks = [
        block for block in model.blocks if hasattr(block, "synapse")
    ]
    return [
        index
        for index, block in enumerate(weighted_blocks)
        if bool(
            getattr(
                block.synapse.pre_hook_fx,
                "is_decomposed_weight_quantizer",
                False,
            )
        )
    ]


def verify(
    checkpoint_path: Path,
    dataset_path: Path,
    sample_count: int,
    atol: float,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Parity requires a self-describing metadata checkpoint")
    if "resolved_config" not in checkpoint:
        raise ValueError("Checkpoint lacks resolved_config")

    config = train_v2.build_config(checkpoint["resolved_config"])
    trainer_model = train_v2.Network(config.model).eval()
    trainer_model.load_state_dict(checkpoint["state_dict"])
    rollout_model, rollout_metadata = load_bootstrap_policy_from_checkpoint(
        checkpoint_path, device="cpu"
    )

    dataset = train_v2.TeacherStudentDataset.from_npz(dataset_path)
    validation_size = max(1, int(len(dataset) * config.training.val_split))
    training_size = len(dataset) - validation_size
    _, validation_set = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    count = min(sample_count, len(validation_set))
    states = torch.stack([validation_set[index][0] for index in range(count)])

    with torch.no_grad():
        trainer_output = train_v2.pure_snn_rate(trainer_model, states)
        rollout_output = rollout_model(states, mode="snn")
    difference = (trainer_output - rollout_output).abs()
    maximum_absolute_error = float(difference.max().item())
    mean_absolute_error = float(difference.mean().item())
    trainer_finite = bool(torch.isfinite(trainer_output).all().item())
    rollout_finite = bool(torch.isfinite(rollout_output).all().item())
    trainer_hooks = selected_hook_indices(trainer_model)
    rollout_hooks = selected_hook_indices(rollout_model)
    diagnostics = []
    for layer_index, block in enumerate(rollout_model.blocks):
        if not hasattr(block, "synapse"):
            continue
        raw = block.synapse.weight.detach()
        hook = block.synapse.pre_hook_fx
        with torch.no_grad():
            quantized = hook(raw) if hook is not None else raw
        hook_diagnostics = dict(getattr(hook, "last_diagnostics", {}))
        diagnostics.append({
            "layer_index": layer_index,
            "selected": bool(getattr(hook, "is_decomposed_weight_quantizer", False)),
            "raw_finite": bool(torch.isfinite(raw).all().item()),
            "quantized_finite": bool(torch.isfinite(quantized).all().item()),
            "saturation_count": int(hook_diagnostics.get("saturation_count", 0)),
        })
    saturation_count = sum(
        int(layer.get("saturation_count", 0))
        for layer in diagnostics
        if layer.get("selected")
    )
    diagnostics_finite = all(
        bool(layer.get("raw_finite"))
        and bool(layer.get("quantized_finite"))
        for layer in diagnostics
        if layer.get("selected")
    )
    expected_selected = (
        list(range(len(diagnostics)))
        if checkpoint["weight_quantization"]["scope"] == "all"
        else [0]
    )
    passed = (
        count > 0
        and trainer_finite
        and rollout_finite
        and diagnostics_finite
        and saturation_count == 0
        and trainer_hooks == expected_selected
        and rollout_hooks == expected_selected
        and maximum_absolute_error <= atol
    )
    return {
        "passed": passed,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "dataset_path": str(dataset_path.resolve()),
        "validation_split_seed": config.training.seed,
        "fixed_batch_selection": "first validation indices after deterministic random_split",
        "sample_count": count,
        "atol": atol,
        "maximum_absolute_error": maximum_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "trainer_output_finite": trainer_finite,
        "rollout_output_finite": rollout_finite,
        "trainer_selected_hook_indices": trainer_hooks,
        "rollout_selected_hook_indices": rollout_hooks,
        "expected_selected_hook_indices": expected_selected,
        "diagnostics_finite": diagnostics_finite,
        "saturation_count": saturation_count,
        "weight_quantization": checkpoint["weight_quantization"],
        "rollout_metadata": rollout_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.atol < 0:
        parser.error("--samples must be positive and --atol non-negative")
    result = verify(args.checkpoint, args.dataset, args.samples, args.atol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
