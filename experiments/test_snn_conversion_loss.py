import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STUDENT_DIR = ROOT / "student"
DEFAULT_DATASET = ROOT / "experiments" / "teacher-student" / "traindata" / "teacher_dataset_100k.npz"
if str(STUDENT_DIR) not in sys.path:
    sys.path.append(str(STUDENT_DIR))

from train_student import (
    SNNConversionConfig,
    SUPPORTED_SNN_READOUTS,
    evaluate_models_against_teacher,
    evaluate_ann_snn_parity,
    load_student_model_from_checkpoint,
    load_teacher_dataset,
    SNNPolicy,
)


def main():
    parser = argparse.ArgumentParser(description="Compare student ANN and converted SNN with parity-first diagnostics.")
    parser.add_argument(
        "--checkpoint",
        default=str(STUDENT_DIR / "student_model_best.pth"),
        help="Path to the student checkpoint to load.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the teacher dataset (.npz with states/actions).",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument("--max-samples", type=int, default=2048, help="Optional cap on the number of dataset samples to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Sigma-delta threshold for conversion.")
    parser.add_argument("--timesteps", type=int, default=3, help="Number of SNN timesteps for conversion.")
    parser.add_argument(
        "--readout",
        choices=SUPPORTED_SNN_READOUTS,
        default="mean",
        help="Readout used for the main SNN output metric.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Optional cap on samples used for parity evaluation. Defaults to all loaded samples.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for ANN inference during the comparison.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    dataset_path = Path(args.dataset).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    device = torch.device(args.device)
    student_model, architecture = load_student_model_from_checkpoint(checkpoint_path, device=device)
    states, teacher_actions = load_teacher_dataset(dataset_path, max_samples=args.max_samples)

    snn_model = SNNPolicy.from_student(
        student_model=student_model,
        conversion_config=SNNConversionConfig(
            threshold=args.threshold,
            timesteps=args.timesteps,
            device=str(device),
            readout=args.readout,
            calibration_samples=args.calibration_samples,
        ),
    )

    parity_metrics = evaluate_ann_snn_parity(
        student_model=student_model,
        snn_model=snn_model,
        states=states,
        batch_size=args.batch_size,
        device=device,
        calibration_samples=args.calibration_samples,
    )
    teacher_metrics = evaluate_models_against_teacher(
        student_model=student_model,
        snn_model=snn_model,
        states=states,
        teacher_actions=teacher_actions,
        batch_size=args.batch_size,
        device=device,
        readout=args.readout,
    )

    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset: {dataset_path}")
    print(f"input_dim: {architecture['input_dim']}")
    print(f"output_dim: {architecture['output_dim']}")
    print(f"hidden_dims: {architecture['hidden_dims']}")
    print(f"samples: {len(states)}")
    print(f"batch_size: {args.batch_size}")
    print(f"threshold: {args.threshold}")
    print(f"timesteps: {args.timesteps}")
    print(f"selected_readout: {parity_metrics['selected_readout']}")
    print(f"parity_relative_l2_error: {parity_metrics['relative_l2_error']:.6f}")
    print(f"parity_mae: {parity_metrics['mae']:.6f}")
    print(f"parity_output_scale_ratio: {parity_metrics['output_scale_ratio']:.6f}")

    for readout, metrics in parity_metrics["readout_diagnostics"].items():
        print(
            f"readout={readout} rel_l2={metrics['relative_l2_error']:.6f} "
            f"mae={metrics['mae']:.6f} scale_ratio={metrics['output_scale_ratio']:.6f}"
        )

    print(f"student_mean_percentage_error: {teacher_metrics['student_mean_percentage_error']:.4f}%")
    print(f"student_median_percentage_error: {teacher_metrics['student_median_percentage_error']:.4f}%")
    print(f"snn_mean_percentage_error: {teacher_metrics['snn_mean_percentage_error']:.4f}%")
    print(f"snn_median_percentage_error: {teacher_metrics['snn_median_percentage_error']:.4f}%")


if __name__ == "__main__":
    main()
