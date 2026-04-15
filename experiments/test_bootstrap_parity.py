import argparse
import json
import sys
from pathlib import Path

sys.path.append("/app/one_policy_to_run_them_all/student")

import torch

from bootstrap_backend import (
    resolve_bootstrap_device,
    BootstrapTrainingConfig,
    BootstrapStudentTrainer,
    evaluate_ann_bootstrap_parity,
    evaluate_bootstrap_against_teacher,
    load_teacher_dataset,
)
from train_student import (
    SNNConversionConfig,
    SNNPolicy,
    evaluate_ann_snn_parity,
    evaluate_models_against_teacher,
    load_student_model_from_checkpoint,
)


def parse_crossover_epochs(raw: str):
    if not raw:
        return ()
    return tuple(int(value) for value in raw.split(",") if value.strip())


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a Lava bootstrap backend against the current ANN+conversion SNN baseline.")
    parser.add_argument("--checkpoint", default="/app/one_policy_to_run_them_all/student/student_model_best.pth")
    parser.add_argument("--dataset", default="/app/one_policy_to_run_them_all/experiments/teacher-student/traindata/teacher_dataset_100k.npz")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--bootstrap-checkpoint-dir", default="/app/one_policy_to_run_them_all/experiments/teacher-student/bootstrap_parity")
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--train-max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-timesteps", type=int, default=3)
    parser.add_argument("--bootstrap-readout", default="mean")
    parser.add_argument("--bootstrap-num-sample-iter", type=int, default=10)
    parser.add_argument("--bootstrap-sample-period", type=int, default=10)
    parser.add_argument("--bootstrap-crossover-epochs", default="")
    parser.add_argument("--bootstrap-neuron-threshold", type=float, default=1.0)
    parser.add_argument("--bootstrap-current-decay", type=float, default=0.25)
    parser.add_argument("--bootstrap-voltage-decay", type=float, default=0.03)
    parser.add_argument("--bootstrap-weight-scale", type=float, default=1.0)
    parser.add_argument("--bootstrap-weight-norm", action="store_true")
    parser.add_argument("--no-bootstrap-init-from-ann", action="store_true")
    parser.add_argument("--conversion-threshold", type=float, default=0.2)
    parser.add_argument("--conversion-timesteps", type=int, default=3)
    parser.add_argument("--conversion-readout", default="mean")
    parser.add_argument("--conversion-output-activation", default="sigma")
    args = parser.parse_args()

    device = resolve_bootstrap_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    output_dir = Path(args.bootstrap_checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    student_model, architecture = load_student_model_from_checkpoint(checkpoint_path, device=device)
    eval_states, eval_actions = load_teacher_dataset(dataset_path, max_samples=args.max_samples)
    train_max_samples = args.train_max_samples if args.train_max_samples > 0 else None
    train_states, train_actions = load_teacher_dataset(dataset_path, max_samples=train_max_samples)

    conversion_snn = SNNPolicy.from_student(
        student_model=student_model,
        conversion_config=SNNConversionConfig(
            threshold=args.conversion_threshold,
            timesteps=args.conversion_timesteps,
            device=str(device),
            readout=args.conversion_readout,
            output_activation=args.conversion_output_activation,
        ),
    )
    conversion_parity = evaluate_ann_snn_parity(
        student_model=student_model,
        snn_model=conversion_snn,
        states=eval_states,
        batch_size=args.batch_size,
        device=device,
    )
    conversion_teacher = evaluate_models_against_teacher(
        student_model=student_model,
        snn_model=conversion_snn,
        states=eval_states,
        teacher_actions=eval_actions,
        batch_size=args.batch_size,
        device=device,
        readout=args.conversion_readout,
    )

    bootstrap_config = BootstrapTrainingConfig(
        dataset_path=str(dataset_path),
        batch_size=args.bootstrap_batch_size,
        learning_rate=1e-4,
        epochs=args.epochs,
        hidden_dims=architecture["hidden_dims"],
        checkpoint_dir=str(output_dir),
        timesteps=args.bootstrap_timesteps,
        readout=args.bootstrap_readout,
        num_sample_iter=args.bootstrap_num_sample_iter,
        sample_period=args.bootstrap_sample_period,
        crossover_epochs=parse_crossover_epochs(args.bootstrap_crossover_epochs),
        neuron_threshold=args.bootstrap_neuron_threshold,
        current_decay=args.bootstrap_current_decay,
        voltage_decay=args.bootstrap_voltage_decay,
        weight_scale=args.bootstrap_weight_scale,
        weight_norm=args.bootstrap_weight_norm,
        initialize_from_ann=not args.no_bootstrap_init_from_ann,
    )
    trainer = BootstrapStudentTrainer(config=bootstrap_config, device=device)
    artifacts = trainer.train_on_arrays(
        states=train_states,
        actions=train_actions,
        initialize_from_ann=None if args.no_bootstrap_init_from_ann else student_model,
    )
    bootstrap_model = trainer.model
    bootstrap_parity = evaluate_ann_bootstrap_parity(
        student_model=student_model,
        bootstrap_model=bootstrap_model,
        states=eval_states,
        batch_size=args.batch_size,
        device=device,
    )
    bootstrap_teacher = evaluate_bootstrap_against_teacher(
        student_model=student_model,
        bootstrap_model=bootstrap_model,
        states=eval_states,
        teacher_actions=eval_actions,
        batch_size=args.batch_size,
        device=device,
    )

    beats_conversion_baseline = (
        bootstrap_parity["relative_l2_error"] < conversion_parity["relative_l2_error"]
        and bootstrap_teacher["snn_mean_percentage_error"] < conversion_teacher["snn_mean_percentage_error"]
        and bootstrap_teacher["student_mean_percentage_error"] <= conversion_teacher["student_mean_percentage_error"] + 1e-6
    )

    report = {
        "student_checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "device": str(device),
        "samples": int(len(eval_states)),
        "student_architecture": architecture,
        "conversion_baseline": {
            "threshold": float(args.conversion_threshold),
            "timesteps": int(args.conversion_timesteps),
            "readout": args.conversion_readout,
            "output_activation": args.conversion_output_activation,
            "parity": conversion_parity,
            "teacher": conversion_teacher,
        },
        "bootstrap": {
            "checkpoint": artifacts.best_checkpoint_path,
            "training": {
                "epochs": int(args.epochs),
                "timesteps": int(args.bootstrap_timesteps),
                "readout": args.bootstrap_readout,
                "num_sample_iter": int(args.bootstrap_num_sample_iter),
                "sample_period": int(args.bootstrap_sample_period),
                "crossover_epochs": list(parse_crossover_epochs(args.bootstrap_crossover_epochs)),
                "neuron_threshold": float(args.bootstrap_neuron_threshold),
                "current_decay": float(args.bootstrap_current_decay),
                "voltage_decay": float(args.bootstrap_voltage_decay),
                "weight_scale": float(args.bootstrap_weight_scale),
                "weight_norm": bool(args.bootstrap_weight_norm),
                "initialized_from_ann": not args.no_bootstrap_init_from_ann,
            },
            "parity": bootstrap_parity,
            "teacher": bootstrap_teacher,
            "best_val_loss": float(artifacts.best_val_loss),
        },
        "bootstrap_beats_conversion_baseline": bool(beats_conversion_baseline),
    }

    print(json.dumps(report, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
