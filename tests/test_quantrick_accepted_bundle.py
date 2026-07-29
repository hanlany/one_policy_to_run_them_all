import hashlib
import json
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bundle(tmp_path):
    checkpoint = tmp_path / "quantrick_network.pt"
    history = tmp_path / "quantrick_training_history.json"
    checkpoint.write_bytes(b"checkpoint fixture")
    history.write_text('{"fixture": true}\n')
    quantization = {
        "mode": "decomposed",
        "target_bits": 24,
        "chunk_bits": 8,
        "sign_mode": "mixed",
        "scope": "all",
    }
    (tmp_path / "acceptance.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "checkpoint_sha256": digest(checkpoint),
                "history_sha256": digest(history),
                "best_full_val_snn_mse": 0.005,
                "mse_gate": 0.00592308546602726,
                "saturation_count": 0,
                "weight_quantization": quantization,
            }
        )
    )
    (tmp_path / "rollout_parity.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checkpoint_sha256": digest(checkpoint),
                "maximum_absolute_error": 0.0,
                "atol": 1e-6,
                "saturation_count": 0,
                "weight_quantization": quantization,
            }
        )
    )
    (tmp_path / "source_label.txt").write_text("fallback_selected\n")
    return checkpoint


def run_verifier(tmp_path, check):
    return subprocess.run(
        [
            sys.executable,
            "dev/codex/scripts/verify_quantrick_accepted_bundle.py",
            str(tmp_path),
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def test_accepted_bundle_verifier_checks_evidence_and_hashes(tmp_path):
    checkpoint = build_bundle(tmp_path)
    run_verifier(tmp_path, check=True)
    result = json.loads((tmp_path / "bundle_verification.json").read_text())
    assert result["verified"] is True
    assert result["source_label"] == "fallback_selected"
    assert result["checkpoint_sha256"] == digest(checkpoint)

    checkpoint.write_bytes(b"tampered")
    failed = run_verifier(tmp_path, check=False)
    assert failed.returncode != 0
    assert "hash differs" in failed.stderr
