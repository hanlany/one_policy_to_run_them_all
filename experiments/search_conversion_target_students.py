import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STUDENT_DIR = ROOT / "student"
DEFAULT_DATASET = ROOT / "experiments" / "teacher-student" / "traindata" / "teacher_dataset_100k.npz"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "teacher-student" / "conversion_target_search"
if str(STUDENT_DIR) not in sys.path:
    sys.path.append(str(STUDENT_DIR))

from train_student import (
    BASELINE_PARITY_REFERENCE,
    CONVERSION_TARGET_HIDDEN_DIMS,
    SUPPORTED_SNN_READOUTS,
    format_hidden_dims,
    search_conversion_target_students,
)


def parse_numeric_list(raw_value: str, cast_type):
    values = [cast_type(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value in the sweep list.")
    return values


def parse_architectures(raw_value: str):
    architectures = []
    for item in raw_value.split(";"):
        item = item.strip()
        if not item:
            continue
        dims = [int(part.strip()) for part in item.split("x") if part.strip()]
        if not dims:
            raise ValueError(f"Could not parse architecture '{item}'. Use forms like 1024x512 or 512x256.")
        architectures.append(dims)
    if not architectures:
        raise ValueError("Expected at least one architecture in the architecture list.")
    return architectures


def main():
    default_architectures = ";".join(format_hidden_dims(hidden_dims) for hidden_dims in CONVERSION_TARGET_HIDDEN_DIMS)

    parser = argparse.ArgumentParser(
        description="Train and rank conversion-target student architectures with ANN-vs-SNN parity as the primary metric."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the teacher dataset (.npz with states/actions).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where candidate checkpoints and the search summary will be written.",
    )
    parser.add_argument(
        "--architectures",
        default=default_architectures,
        help="Semicolon-separated hidden-dim candidates, for example '1024x512;512x512;512x256'.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.05,0.1,0.2",
        help="Comma-separated threshold sweep values.",
    )
    parser.add_argument(
        "--timesteps",
        default="5,10,20",
        help="Comma-separated timestep sweep values.",
    )
    parser.add_argument(
        "--readout",
        choices=SUPPORTED_SNN_READOUTS,
        default="mean",
        help="Primary readout used to score conversion parity.",
    )
    parser.add_argument("--train-batch-size", type=int, default=64, help="Batch size used while training each student candidate.")
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=128,
        help="Batch size used for ANN-SNN parity and teacher-label evaluation.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for candidate student training.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs for each candidate student.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2048,
        help="Number of real states used for parity and teacher-label evaluation.",
    )
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=None,
        help="Optional cap on training samples. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Optional cap on samples used by the parity evaluator.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split used while training each candidate student.",
    )
    parser.add_argument(
        "--student-error-tolerance",
        type=float,
        default=0.10,
        help="Allowed relative regression on student mean teacher error versus the current baseline.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for training and parity evaluation.",
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the current 5x1024 student as a baseline candidate.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    architectures = parse_architectures(args.architectures)
    thresholds = parse_numeric_list(args.thresholds, float)
    timesteps = parse_numeric_list(args.timesteps, int)

    results = search_conversion_target_students(
        dataset_path=dataset_path,
        output_dir=output_dir,
        candidate_architectures=architectures,
        threshold_values=thresholds,
        timestep_values=timesteps,
        train_batch_size=args.train_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_samples=args.max_samples,
        train_max_samples=args.train_max_samples,
        calibration_samples=args.calibration_samples,
        readout=args.readout,
        device=args.device,
        val_split=args.val_split,
        include_baseline=args.include_baseline,
        baseline_reference=BASELINE_PARITY_REFERENCE,
        student_error_regression_tolerance=args.student_error_tolerance,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "search_results.json"
    summary_path.write_text(json.dumps(results, indent=2))

    print(f"dataset: {dataset_path}")
    print(f"output_dir: {output_dir}")
    print(f"summary_path: {summary_path}")
    print(f"device: {results['device']}")
    print(f"evaluation_samples: {results['evaluation_samples']}")
    print(f"train_samples: {results['train_samples']}")
    print(f"baseline_relative_l2_error: {results['baseline_reference']['relative_l2_error']:.4f}")
    print(f"baseline_output_scale_ratio: {results['baseline_reference']['output_scale_ratio']:.4f}")
    print(f"baseline_student_error_threshold: {results['baseline_reference']['student_error_threshold']:.4f}%")
    print(f"snn_teacher_error_ratio_limit: {results['baseline_reference']['snn_teacher_error_ratio_limit']:.4f}")
    print(f"snn_mean_percentage_error_limit: {results['baseline_reference']['snn_mean_percentage_error_limit']:.4f}%")

    print("ranked_results:")
    for rank, result in enumerate(results["results"], start=1):
        best_conversion = result["best_conversion"]
        activation_stats = result["activation_stats"]
        hidden_linear_ratios = ", ".join(
            f"{stat['abs_max_ratio_to_io']:.2f}" for stat in activation_stats["hidden_linear_stats"]
        )
        status = []
        if result["passes_baseline_thresholds"]:
            status.append("accepted")
        else:
            status.append("candidate")
        if not result["passes_snn_quality_gate"]:
            status.append("quality_failed")
        if result["activation_scale_rejected"]:
            status.append("activation_rejected")
        print(
            f"  rank={rank} arch={result['architecture_name']} status={','.join(status)} "
            f"threshold={best_conversion['threshold']:.2f} timesteps={best_conversion['timesteps']} "
            f"rel_l2={best_conversion['parity_relative_l2_error']:.6f} "
            f"scale_ratio={best_conversion['parity_output_scale_ratio']:.6f} "
            f"student_mean_pct={best_conversion['student_mean_percentage_error']:.4f}% "
            f"snn_mean_pct={best_conversion['snn_mean_percentage_error']:.4f}% "
            f"snn_quality_threshold={best_conversion['snn_quality_threshold']:.4f}%"
        )
        print(
            f"    checkpoint={result['best_checkpoint_path']} "
            f"hidden_linear_abs_max_ratios=[{hidden_linear_ratios}]"
        )

    chosen_result = results["best_accepted_result"] or results["best_result"]
    chosen_label = "best_accepted_result" if results["best_accepted_result"] is not None else "best_result"
    best_conversion = chosen_result["best_conversion"]
    print(f"{chosen_label}:")
    print(f"  arch: {chosen_result['architecture_name']}")
    print(f"  checkpoint: {chosen_result['best_checkpoint_path']}")
    print(f"  threshold: {best_conversion['threshold']}")
    print(f"  timesteps: {best_conversion['timesteps']}")
    print(f"  parity_relative_l2_error: {best_conversion['parity_relative_l2_error']:.6f}")
    print(f"  parity_output_scale_ratio: {best_conversion['parity_output_scale_ratio']:.6f}")
    print(f"  student_mean_percentage_error: {best_conversion['student_mean_percentage_error']:.4f}%")
    print(f"  snn_mean_percentage_error: {best_conversion['snn_mean_percentage_error']:.4f}%")
    print(f"  snn_quality_threshold: {best_conversion['snn_quality_threshold']:.4f}%")
    print(f"  passes_snn_quality_gate: {best_conversion['passes_snn_quality_gate']}")


if __name__ == "__main__":
    main()
