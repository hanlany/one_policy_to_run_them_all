#!/usr/bin/env python3
"""Validate a completed QuanTrick training result against the hard gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


MSE_GATE = 0.00592308546602726


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(history_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    history = json.loads(history_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Final QuanTrick checkpoint must be a metadata payload")

    quantization = checkpoint.get("weight_quantization")
    if not isinstance(quantization, dict):
        raise ValueError("Checkpoint lacks weight_quantization metadata")
    if (
        quantization.get("mode") != "decomposed"
        or int(quantization.get("target_bits", 0)) != 24
        or int(quantization.get("chunk_bits", 0)) != 8
    ):
        raise ValueError(f"Checkpoint is not decomposed 24/8: {quantization}")

    diagnostics = history["history"]["quantization_diagnostics"][-1]
    if diagnostics.get("stage") != "final_best_checkpoint":
        raise ValueError("History lacks final-best checkpoint diagnostics")
    layers = diagnostics["layers"]
    selected_layers = [layer for layer in layers if layer.get("selected")]
    saturation_count = sum(
        int(layer.get("saturation_count", 0)) for layer in selected_layers
    )
    diagnostics_finite = bool(selected_layers) and all(
        bool(layer.get("raw_finite"))
        and bool(layer.get("quantized_finite"))
        for layer in selected_layers
    )
    state_finite = all(
        bool(torch.isfinite(value).all().item())
        for value in checkpoint["state_dict"].values()
        if torch.is_tensor(value)
    )

    mse = float(history["best_full_val_snn_mse"])
    mean_pct = float(history["best_full_val_mean_percentage_error"])
    median_pct = float(history["best_full_val_median_percentage_error"])
    metrics_finite = all(math.isfinite(value) for value in (mse, mean_pct, median_pct))
    accepted = (
        metrics_finite
        and diagnostics_finite
        and state_finite
        and saturation_count == 0
        and mse <= MSE_GATE
    )
    return {
        "accepted": accepted,
        "mse_gate": MSE_GATE,
        "best_epoch": int(history["best_epoch"]),
        "best_full_val_snn_mse": mse,
        "best_full_val_mean_percentage_error": mean_pct,
        "best_full_val_median_percentage_error": median_pct,
        "metrics_finite": metrics_finite,
        "diagnostics_finite": diagnostics_finite,
        "checkpoint_state_finite": state_finite,
        "selected_decomposed_synapses": len(selected_layers),
        "saturation_count": saturation_count,
        "weight_quantization": quantization,
        "history_path": str(history_path.resolve()),
        "history_sha256": sha256(history_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.history, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
