#!/usr/bin/env python3
"""Verify the stable accepted QuanTrick bundle before recording."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_SOURCE_LABELS = {
    "full_selected",
    "fallback_selected",
    "fallback_runner_up",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def verify(bundle_dir: Path) -> dict[str, Any]:
    checkpoint_path = bundle_dir / "quantrick_network.pt"
    history_path = bundle_dir / "quantrick_training_history.json"
    acceptance_path = bundle_dir / "acceptance.json"
    parity_path = bundle_dir / "rollout_parity.json"
    source_label_path = bundle_dir / "source_label.txt"
    for path in (
        checkpoint_path,
        history_path,
        acceptance_path,
        parity_path,
        source_label_path,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Accepted bundle file is missing or empty: {path}")

    acceptance = load_json(acceptance_path)
    parity = load_json(parity_path)
    source_label = source_label_path.read_text().strip()
    if source_label not in ALLOWED_SOURCE_LABELS:
        raise ValueError(f"Unexpected accepted source label: {source_label!r}")
    if acceptance.get("accepted") is not True:
        raise ValueError("Accepted bundle accuracy evidence is not accepted=true")
    if parity.get("passed") is not True:
        raise ValueError("Accepted bundle rollout parity evidence is not passed=true")

    checkpoint_hash = sha256(checkpoint_path)
    history_hash = sha256(history_path)
    if acceptance.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Promoted checkpoint hash differs from acceptance evidence")
    if parity.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Promoted checkpoint hash differs from parity evidence")
    if acceptance.get("history_sha256") != history_hash:
        raise ValueError("Promoted history hash differs from acceptance evidence")

    quantization = acceptance.get("weight_quantization")
    if quantization != parity.get("weight_quantization"):
        raise ValueError("Acceptance/parity quantization metadata differs")
    if not isinstance(quantization, dict) or (
        quantization.get("mode") != "decomposed"
        or quantization.get("target_bits") != 24
        or quantization.get("chunk_bits") != 8
    ):
        raise ValueError(f"Accepted bundle is not decomposed 24/8: {quantization}")
    if float(parity.get("maximum_absolute_error", float("inf"))) > float(
        parity.get("atol", -1)
    ):
        raise ValueError("Accepted bundle exceeds its rollout parity tolerance")
    if int(acceptance.get("saturation_count", -1)) != 0 or int(
        parity.get("saturation_count", -1)
    ) != 0:
        raise ValueError("Accepted bundle reports quantization saturation")

    return {
        "verified": True,
        "source_label": source_label,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "history_path": str(history_path.resolve()),
        "history_sha256": history_hash,
        "best_full_val_snn_mse": acceptance["best_full_val_snn_mse"],
        "mse_gate": acceptance["mse_gate"],
        "parity_maximum_absolute_error": parity["maximum_absolute_error"],
        "parity_atol": parity["atol"],
        "weight_quantization": quantization,
        "saturation_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = verify(args.bundle_dir)
    output = args.output or args.bundle_dir / "bundle_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
