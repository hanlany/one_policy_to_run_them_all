#!/usr/bin/env python3
"""Rank the matched QuanTrick candidates from saved training histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CANDIDATES = (
    ("legacy_8bit", "legacy_8bit"),
    ("decomposed_first", "decomposed_first"),
    ("decomposed_all", "decomposed_all"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(run_root: Path, label: str, directory: str) -> dict[str, Any]:
    candidate_dir = run_root / "candidates" / directory
    history_path = candidate_dir / f"{label}_training_history.json"
    checkpoint_path = candidate_dir / f"{label}_network.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Candidate {label} is incomplete: expected {history_path} "
            f"and {checkpoint_path}"
        )

    payload = json.loads(history_path.read_text())
    diagnostics = payload["history"]["quantization_diagnostics"][-1]
    if diagnostics.get("stage") != "final_best_checkpoint":
        raise ValueError(
            f"Candidate {label} lacks final-best quantization diagnostics"
        )
    layers = diagnostics["layers"]
    saturation_count = sum(
        int(layer.get("saturation_count", 0)) for layer in layers
    )
    finite = all(
        bool(layer.get("raw_finite"))
        and bool(layer.get("quantized_finite"))
        for layer in layers
    )
    mse = float(payload["best_full_val_snn_mse"])
    mean_pct = float(payload["best_full_val_mean_percentage_error"])
    median_pct = float(payload["best_full_val_median_percentage_error"])
    metrics_finite = all(
        math.isfinite(value) for value in (mse, mean_pct, median_pct)
    )

    return {
        "label": label,
        "mode": payload["config"]["model"]["weight_quantization"]["mode"],
        "scope": payload["config"]["model"]["weight_quantization"]["scope"],
        "best_epoch": int(payload["best_epoch"]),
        "best_full_val_snn_mse": mse,
        "best_full_val_mean_percentage_error": mean_pct,
        "best_full_val_median_percentage_error": median_pct,
        "metrics_finite": metrics_finite,
        "weights_and_outputs_finite": finite,
        "saturation_count": saturation_count,
        "eligible": metrics_finite and finite and saturation_count == 0,
        "history_path": str(history_path),
        "history_sha256": sha256(history_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
    }


def rank_candidates(run_root: Path) -> dict[str, Any]:
    candidates = [
        load_candidate(run_root, label, directory)
        for label, directory in CANDIDATES
    ]
    decomposed = [
        candidate
        for candidate in candidates
        if candidate["mode"] == "decomposed" and candidate["eligible"]
    ]
    decomposed.sort(
        key=lambda candidate: (
            candidate["best_full_val_snn_mse"],
            candidate["best_full_val_median_percentage_error"],
            candidate["best_full_val_mean_percentage_error"],
        )
    )
    if len(decomposed) != 2:
        raise ValueError(
            "Expected two eligible decomposed candidates, found "
            f"{len(decomposed)}"
        )

    return {
        "selection_order": [
            "best_full_val_snn_mse",
            "best_full_val_median_percentage_error",
            "best_full_val_mean_percentage_error",
        ],
        "candidates": candidates,
        "decomposed_ranking": [
            candidate["label"] for candidate in decomposed
        ],
        "selected": decomposed[0]["label"],
        "runner_up": decomposed[1]["label"],
        "rationale": (
            "Lowest full-validation MSE; exact ties break by median "
            "percentage error, then mean percentage error."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--output", type=Path, default=None
    )
    args = parser.parse_args()
    result = rank_candidates(args.run_root)
    output = args.output or args.run_root / "candidate_ranking.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved ranking to {output}")


if __name__ == "__main__":
    main()

