#!/usr/bin/env python3
"""Run offline SNN architecture and bootstrap scheduler sweeps.

This runner follows the durable offline-orchestrator pattern used by
``run_snn_points_1_2_sweep.py`` while sweeping two higher-level surfaces:

1. Bootstrap network architecture with fixed long-best SNN params.
2. Lava ``bootstrap.routine.Scheduler`` settings using the best architecture.

Each child training run is isolated into a labeled output directory with a
command manifest, per-trial log, result JSON, and resumable status summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SNN_DIR = REPO_ROOT / "snn"
DEFAULT_EXPERIMENT_ROOT = SNN_DIR / "experiments" / "snn_v1_arch_scheduler"
DEFAULT_CONFIG = SNN_DIR / "train_v2_config_long.yaml"
DEFAULT_TRAIN_SCRIPT = SNN_DIR / "train_v2.py"
DEFAULT_DATASET = (
    REPO_ROOT
    / "experiments"
    / "teacher-student"
    / "online_dagger_snnv2_longbest"
    / "teacher_student_dagger_dataset.npz"
)
DEFAULT_CHECKPOINT = DEFAULT_DATASET.with_name("student_model_latest.pth")

FIXED_SNN_PARAMS = {
    "time_steps": 20,
    "input_strategy": "signed_split",
    "input_weight": 1.0,
    "input_bias": 0.0,
    "threshold": 0.2,
    "current_decay": 0.3,
    "voltage_decay": 0.02,
}
DEFAULT_ARCHITECTURES = [
    [1024, 1024, 1024, 1024, 1024],
    [1024, 1024, 1024, 1024, 1024, 1024],
    [1536, 1536, 1536, 1536, 1536],
    [2048, 2048, 2048, 2048],
    [2048, 2048, 2048, 2048, 2048],
    [2048, 2048, 1536, 1536, 1024],
    [1024, 1536, 2048, 1536, 1024],
    [1536, 1536, 1024, 1024, 1024, 1024],
]

METRIC_LINE_RE = re.compile(
    r"\[Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+)\]\s+"
    r"train_snn_mse=(?P<train>[^\s]+)\s+"
    r"val_subset_snn_mse=(?P<val_subset>[^\s]+)\s+"
    r"full_val_snn_mse=(?P<full_val>[^\s]+)\s+"
    r"best_snn_val_mse=(?P<best>[^\s]+).*?"
    r"full_val_mean_pct=(?P<full_val_mean_pct>[^\s]+)\s+"
    r"full_val_median_pct=(?P<full_val_median_pct>[^\s]+)\s+"
    r"lr=(?P<lr>[^\s]+)\s+"
    r"output_abs_mean=(?P<output_abs_mean>[^\s]+)"
)

RESULT_JSON_NAMES = (
    "snn_training_history.json",
    "results.json",
    "training_results.json",
    "training_summary.json",
    "metrics.json",
    "summary.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        text = raw.strip().removesuffix("%")
        if text.lower() == "n/a":
            return None
        try:
            value = float(text)
        except ValueError:
            return None
    else:
        return None
    if value in {float("inf"), float("-inf")} or value != value:
        return None
    return value


def format_value(value: int | float | str) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_").replace(",", "-")


def format_arch(hidden_dims: list[int]) -> str:
    return "x".join(str(int(dim)) for dim in hidden_dims)


def format_hidden_dims(hidden_dims: list[int]) -> str:
    return "[" + ",".join(str(int(dim)) for dim in hidden_dims) + "]"


def parse_architectures(raw: str | None) -> list[list[int]]:
    if not raw:
        return [list(item) for item in DEFAULT_ARCHITECTURES]
    architectures: list[list[int]] = []
    for item in raw.split(";"):
        dims = [int(value.strip()) for value in item.split(",") if value.strip()]
        if not dims:
            raise argparse.ArgumentTypeError(f"Invalid empty architecture in {raw!r}")
        architectures.append(dims)
    return architectures


def parse_int_csv(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"Expected at least one integer in {raw!r}")
    return values


def parse_crossover(raw: str) -> list[int]:
    if raw.strip() in {"", "[]", "none", "None"}:
        return []
    return parse_int_csv(raw)


def parse_crossover_list(raw: str) -> list[list[int]]:
    groups: list[list[int]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        groups.append(parse_crossover(item))
    return groups


def arch_label(prefix: str, hidden_dims: list[int]) -> str:
    return f"{prefix}_h{format_arch(hidden_dims)}"


def scheduler_label(prefix: str, trial: dict[str, Any]) -> str:
    if trial["mode"] == "pure_snn":
        return f"{prefix}_pure_snn"
    crossover = trial.get("crossover_epochs") or []
    cross_text = "none" if not crossover else "c" + "-".join(str(int(epoch)) for epoch in crossover)
    return f"{prefix}_ni{trial['num_sample_iter']}_sp{trial['sample_period']}_{cross_text}"


def parse_metric_line(line: str) -> dict[str, Any] | None:
    match = METRIC_LINE_RE.search(line)
    if match is None:
        return None
    data = match.groupdict()
    return {
        "epoch": int(data["epoch"]),
        "epochs": int(data["epochs"]),
        "train_snn_mse": parse_float(data["train"]),
        "val_subset_snn_mse": parse_float(data["val_subset"]),
        "full_val_snn_mse": parse_float(data["full_val"]),
        "best_snn_val_mse": parse_float(data["best"]),
        "full_val_mean_percentage_error": parse_float(data["full_val_mean_pct"]),
        "full_val_median_percentage_error": parse_float(data["full_val_median_pct"]),
        "learning_rate": parse_float(data["lr"]),
        "output_abs_mean": parse_float(data["output_abs_mean"]),
    }


def non_none(values: list[Any]) -> list[tuple[int, Any]]:
    return [(index, value) for index, value in enumerate(values) if value is not None]


def value_at(history: dict[str, Any], key: str, index: int | None) -> Any:
    if index is None:
        return None
    values = history.get(key, [])
    if index >= len(values):
        return None
    return values[index]


def last_value(history: dict[str, Any], key: str) -> Any:
    values = non_none(history.get(key, []))
    return values[-1][1] if values else None


def summarize_history_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    history = payload.get("history", payload)
    best_mse = parse_float(payload.get("best_full_val_snn_mse", payload.get("best_snn_val_loss")))
    best_epoch = payload.get("best_epoch")
    best_index = int(best_epoch) - 1 if best_epoch is not None else None
    if best_mse is None:
        full_vals = [(index, parse_float(value)) for index, value in non_none(history.get("full_val_snn_loss", []))]
        full_vals = [(index, value) for index, value in full_vals if value is not None]
        if full_vals:
            best_index, best_mse = min(full_vals, key=lambda item: item[1])
            best_epoch = best_index + 1
    return {
        "best_full_val_snn_mse": best_mse,
        "best_epoch": best_epoch,
        "best_full_val_mean_percentage_error": parse_float(
            payload.get("best_full_val_mean_percentage_error", value_at(history, "full_val_mean_percentage_error", best_index))
        ),
        "best_full_val_median_percentage_error": parse_float(
            payload.get("best_full_val_median_percentage_error", value_at(history, "full_val_median_percentage_error", best_index))
        ),
        "final_full_val_mean_percentage_error": parse_float(last_value(history, "full_val_mean_percentage_error")),
        "final_full_val_median_percentage_error": parse_float(last_value(history, "full_val_median_percentage_error")),
        "history": history,
    }


def discover_result_summary(output_dir: Path) -> dict[str, Any]:
    candidates = [output_dir / filename for filename in RESULT_JSON_NAMES]
    candidates.extend(sorted(output_dir.glob("*history*.json")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        payload = load_json(path)
        if payload is None:
            continue
        summary = summarize_history_payload(payload)
        if summary.get("best_full_val_snn_mse") is not None:
            summary["metric_source"] = str(path)
            return summary
    return {}


def selection_key(result: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(result.get("best_full_val_snn_mse") if result.get("best_full_val_snn_mse") is not None else float("inf")),
        float(
            result.get("best_full_val_mean_percentage_error")
            if result.get("best_full_val_mean_percentage_error") is not None
            else float("inf")
        ),
        float(
            result.get("best_full_val_median_percentage_error")
            if result.get("best_full_val_median_percentage_error") is not None
            else float("inf")
        ),
        str(result.get("label", "")),
    )


def successful_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [result for result in results if result.get("exit_code") == 0 and result.get("best_full_val_snn_mse") is not None],
        key=selection_key,
    )


def build_command(*, python: str, train_script: Path, config: Path, overrides: list[str]) -> list[str]:
    command = [python, str(train_script), "--config", str(config)]
    for override in overrides:
        command.extend(["--set", override])
    return command


def fixed_overrides(
    *,
    dataset: Path,
    checkpoint: Path,
    output_dir: Path,
    label: str,
    hidden_dims: list[int],
    epochs: int,
    full_val_interval: int,
    training_mode: str,
    num_sample_iter: int | None,
    sample_period: int | None,
    crossover_epochs: list[int],
    export_hdf5: bool,
    save_plot: bool,
    device: str | None,
) -> list[str]:
    overrides = [
        f"paths.dataset={dataset}",
        f"paths.ann_checkpoint={checkpoint}",
        f"paths.output_dir={output_dir}",
        f"paths.checkpoint_name={label}_network.pt",
        f"paths.export_name={label}_network.net",
        f"paths.plot_name={label}_training_curves.png",
        f"paths.history_name={label}_training_history.json",
        f"model.hidden_dims={format_hidden_dims(hidden_dims)}",
        "model.init_policy=partial",
        f"model.time_steps={FIXED_SNN_PARAMS['time_steps']}",
        f"model.input_strategy={FIXED_SNN_PARAMS['input_strategy']}",
        f"model.input_weight={FIXED_SNN_PARAMS['input_weight']}",
        f"model.input_bias={FIXED_SNN_PARAMS['input_bias']}",
        f"model.neuron.threshold={FIXED_SNN_PARAMS['threshold']}",
        f"model.neuron.current_decay={FIXED_SNN_PARAMS['current_decay']}",
        f"model.neuron.voltage_decay={FIXED_SNN_PARAMS['voltage_decay']}",
        f"bootstrap_training.mode={training_mode}",
        f"bootstrap_training.crossover_epochs={format_hidden_dims(crossover_epochs)}",
        f"training.epochs={epochs}",
        f"training.full_val_interval={full_val_interval}",
        f"runtime.export_hdf5={str(export_hdf5).lower()}",
        f"runtime.save_plot={str(save_plot).lower()}",
    ]
    if num_sample_iter is not None:
        overrides.append(f"bootstrap_training.num_sample_iter={num_sample_iter}")
    if sample_period is not None:
        overrides.append(f"bootstrap_training.sample_period={sample_period}")
    if device:
        overrides.append(f"runtime.device={device}")
    return overrides


def append_sample_caps(overrides: list[str], args: argparse.Namespace) -> None:
    if args.short_max_train_samples is not None:
        overrides.append(f"training.max_train_samples={args.short_max_train_samples}")
    if args.short_max_val_samples is not None:
        overrides.append(f"training.max_val_samples={args.short_max_val_samples}")
    if args.short_val_eval_samples is not None:
        overrides.append(f"training.val_eval_samples={args.short_val_eval_samples}")


def run_training(
    *,
    phase: str,
    label: str,
    params: dict[str, Any],
    output_dir: Path,
    log_path: Path,
    command: list[str],
    resume: bool,
) -> dict[str, Any]:
    result_path = output_dir / "trial_result.json"
    existing = load_json(result_path) if resume and result_path.exists() else None
    if existing and existing.get("exit_code") == 0 and existing.get("best_full_val_snn_mse") is not None:
        print(f"[resume] {phase} {label}: using existing result {result_path}", flush=True)
        existing["resumed"] = True
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = utc_now()
    metadata = {
        "phase": phase,
        "label": label,
        "params": params,
        "output_dir": output_dir,
        "log_path": log_path,
        "command": command,
        "command_pretty": " ".join(shlex.quote(part) for part in command),
        "started_at": start_time,
    }
    write_json(output_dir / "command.json", metadata)

    print(f"[start] {phase} {label}", flush=True)
    print(f"[log] {log_path}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")

    epoch_metrics: list[dict[str, Any]] = []
    process_start = time.monotonic()
    best_full_val_mse = None
    best_epoch = None
    best_mean_pct = None
    best_median_pct = None
    last_full_val_mse = None

    with log_path.open("w", buffering=1) as log_file:
        log_file.write(f"# started_at: {start_time}\n")
        log_file.write(f"# cwd: {REPO_ROOT}\n")
        log_file.write("# command: " + " ".join(shlex.quote(part) for part in command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            print(line, end="", flush=True)
            metrics = parse_metric_line(line)
            if metrics is None:
                continue
            epoch_metrics.append(metrics)
            full_val = metrics["full_val_snn_mse"]
            if full_val is not None:
                last_full_val_mse = full_val
                if best_full_val_mse is None or full_val < best_full_val_mse:
                    best_full_val_mse = full_val
                    best_epoch = metrics["epoch"]
                    best_mean_pct = metrics["full_val_mean_percentage_error"]
                    best_median_pct = metrics["full_val_median_percentage_error"]
        exit_code = process.wait()

    summary = discover_result_summary(output_dir)
    if summary.get("best_full_val_snn_mse") is not None:
        best_full_val_mse = summary["best_full_val_snn_mse"]
        best_epoch = summary.get("best_epoch", best_epoch)
        best_mean_pct = summary.get("best_full_val_mean_percentage_error", best_mean_pct)
        best_median_pct = summary.get("best_full_val_median_percentage_error", best_median_pct)

    result = {
        **metadata,
        "finished_at": utc_now(),
        "duration_seconds": round(time.monotonic() - process_start, 3),
        "exit_code": exit_code,
        "best_full_val_snn_mse": best_full_val_mse,
        "best_epoch": best_epoch,
        "best_full_val_mean_percentage_error": best_mean_pct,
        "best_full_val_median_percentage_error": best_median_pct,
        "final_full_val_mean_percentage_error": summary.get("final_full_val_mean_percentage_error"),
        "final_full_val_median_percentage_error": summary.get("final_full_val_median_percentage_error"),
        "last_full_val_snn_mse": last_full_val_mse,
        "metric_source": summary.get("metric_source", "stdout"),
        "checkpoint_path": output_dir / f"{label}_network.pt",
        "export_path": output_dir / f"{label}_network.net",
        "plot_path": output_dir / f"{label}_training_curves.png",
        "epoch_metrics": epoch_metrics,
    }
    write_json(result_path, result)

    if exit_code != 0:
        print(f"[failed] {phase} {label}: exit_code={exit_code}", flush=True)
    elif best_full_val_mse is None:
        print(f"[warning] {phase} {label}: no full-validation MSE found", flush=True)
    else:
        print(
            f"[done] {phase} {label}: best_full_val_snn_mse={best_full_val_mse:.9f} "
            f"mean_pct={best_mean_pct} median_pct={best_median_pct}",
            flush=True,
        )
    return result


def build_arch_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    trials = [
        {
            "phase": "architecture",
            "hidden_dims": list(hidden_dims),
            "training_mode": "pure_snn",
            "num_sample_iter": None,
            "sample_period": None,
            "crossover_epochs": [],
        }
        for hidden_dims in args.architectures
    ]
    if args.max_arch_trials is not None:
        trials = trials[: args.max_arch_trials]
    return trials


def build_scheduler_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = [
        {
            "phase": "scheduler",
            "mode": "pure_snn",
            "training_mode": "pure_snn",
            "num_sample_iter": None,
            "sample_period": None,
            "crossover_epochs": [],
        }
    ]
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for num_sample_iter in args.scheduler_num_sample_iters:
        for sample_period in args.scheduler_sample_periods:
            key = (num_sample_iter, sample_period, ())
            seen.add(key)
            trials.append(
                {
                    "phase": "scheduler",
                    "mode": "scheduler",
                    "training_mode": "scheduler",
                    "num_sample_iter": num_sample_iter,
                    "sample_period": sample_period,
                    "crossover_epochs": [],
                }
            )
    for num_sample_iter, sample_period in [(5, 5), (10, 10), (20, 10)]:
        for crossover_epochs in args.scheduler_crossover_probes:
            key = (num_sample_iter, sample_period, tuple(crossover_epochs))
            if key in seen:
                continue
            seen.add(key)
            trials.append(
                {
                    "phase": "scheduler",
                    "mode": "scheduler",
                    "training_mode": "scheduler",
                    "num_sample_iter": num_sample_iter,
                    "sample_period": sample_period,
                    "crossover_epochs": list(crossover_epochs),
                }
            )
    if args.max_scheduler_trials is not None:
        trials = trials[: args.max_scheduler_trials]
    return trials


def write_phase_results(run_root: Path, phase: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = successful_results(results)
    write_json(
        run_root / f"{phase}_sweep_results.json",
        {
            "updated_at": utc_now(),
            "phase": phase,
            "run_root": run_root,
            "total_planned_trials": len(results),
            "completed_trials": len(results),
            "successful_trials": len(ranked),
            "selection_metric": "best_full_val_snn_mse, then best_full_val_mean_percentage_error, then best_full_val_median_percentage_error",
            "best_result": ranked[0] if ranked else None,
            "ranked_results": ranked,
            "results": results,
        },
    )
    return ranked


def write_csv_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "phase",
        "label",
        "hidden_dims",
        "training_mode",
        "num_sample_iter",
        "sample_period",
        "crossover_epochs",
        "best_full_val_snn_mse",
        "best_epoch",
        "best_full_val_mean_percentage_error",
        "best_full_val_median_percentage_error",
        "duration_seconds",
        "exit_code",
        "output_dir",
        "log_path",
    ]
    ranked = successful_results(rows)
    rank_by_label = {row["label"]: rank for rank, row in enumerate(ranked, start=1)}
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: rank_by_label.get(item.get("label"), 10_000)):
            params = row.get("params", {})
            writer.writerow(
                {
                    "rank": rank_by_label.get(row.get("label"), ""),
                    "phase": row.get("phase", ""),
                    "label": row.get("label", ""),
                    "hidden_dims": format_arch(params.get("hidden_dims", [])) if params.get("hidden_dims") else "",
                    "training_mode": params.get("training_mode", ""),
                    "num_sample_iter": params.get("num_sample_iter", ""),
                    "sample_period": params.get("sample_period", ""),
                    "crossover_epochs": params.get("crossover_epochs", ""),
                    "best_full_val_snn_mse": row.get("best_full_val_snn_mse", ""),
                    "best_epoch": row.get("best_epoch", ""),
                    "best_full_val_mean_percentage_error": row.get("best_full_val_mean_percentage_error", ""),
                    "best_full_val_median_percentage_error": row.get("best_full_val_median_percentage_error", ""),
                    "duration_seconds": row.get("duration_seconds", ""),
                    "exit_code": row.get("exit_code", ""),
                    "output_dir": row.get("output_dir", ""),
                    "log_path": row.get("log_path", ""),
                }
            )


def write_plots(analysis_dir: Path, arch_results: list[dict[str, Any]], scheduler_results: list[dict[str, Any]], long_results: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis_dir.mkdir(parents=True, exist_ok=True)

    def bar_plot(rows: list[dict[str, Any]], title: str, filename: str) -> None:
        ranked = successful_results(rows)
        if not ranked:
            return
        labels = [row["label"].replace("arch_", "").replace("sched_", "") for row in ranked]
        values = [float(row["best_full_val_snn_mse"]) for row in ranked]
        fig, ax = plt.subplots(figsize=(max(10, len(ranked) * 0.7), 5.5))
        bars = ax.bar(range(len(ranked)), values, color="#1f77b4")
        ax.set_title(title)
        ax.set_ylabel("Best full-val SNN MSE")
        ax.set_xticks(range(len(ranked)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for bar, row in zip(bars, ranked):
            mean_pct = row.get("best_full_val_mean_percentage_error")
            median_pct = row.get("best_full_val_median_percentage_error")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.4g}\nmean {mean_pct if mean_pct is not None else 'n/a'}\nmed {median_pct if median_pct is not None else 'n/a'}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.margins(y=0.25)
        fig.tight_layout()
        fig.savefig(analysis_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    bar_plot(arch_results, "Architecture Sweep Ranking", "architecture_sweep_ranking.png")
    bar_plot(scheduler_results, "Scheduler Sweep Ranking", "scheduler_sweep_ranking.png")
    if long_results:
        bar_plot(long_results, "Long Confirmation Runs", "long_confirmation_ranking.png")


def write_analysis(run_root: Path, arch_results: list[dict[str, Any]], scheduler_results: list[dict[str, Any]], long_results: list[dict[str, Any]]) -> None:
    analysis_dir = run_root / "analysis"
    write_csv_summary(analysis_dir / "architecture_sweep_summary.csv", arch_results)
    write_csv_summary(analysis_dir / "scheduler_sweep_summary.csv", scheduler_results)
    write_csv_summary(analysis_dir / "long_confirmation_summary.csv", long_results)
    write_plots(analysis_dir, arch_results, scheduler_results, long_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline SNN architecture and scheduler sweeps.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to invoke snn/train_v2.py.")
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT, help="Path to snn/train_v2.py.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base train_v2 YAML config.")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT, help="Parent directory for run IDs.")
    parser.add_argument("--run-root", type=Path, default=None, help="Concrete run directory. Defaults to experiment-root/run-id.")
    parser.add_argument("--run-id", default=None, help="Run ID used when --run-root is omitted.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Teacher-student dataset path.")
    parser.add_argument("--ann-checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="ANN checkpoint used for partial initialization.")
    parser.add_argument("--device", default=None, help="Optional runtime.device override, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--architectures", type=parse_architectures, default=parse_architectures(None), help="Semicolon-separated architectures, e.g. '1024,1024;2048,2048,2048'.")
    parser.add_argument("--scheduler-num-sample-iters", type=parse_int_csv, default=parse_int_csv("1,5,10,20"))
    parser.add_argument("--scheduler-sample-periods", type=parse_int_csv, default=parse_int_csv("1,5,10,20"))
    parser.add_argument("--scheduler-crossover-probes", type=parse_crossover_list, default=parse_crossover_list("10,20,30"))
    parser.add_argument("--short-epochs", type=int, default=40)
    parser.add_argument("--long-epochs", type=int, default=300)
    parser.add_argument("--short-full-val-interval", type=int, default=10)
    parser.add_argument("--long-full-val-interval", type=int, default=10)
    parser.add_argument("--short-max-train-samples", type=int, default=50000)
    parser.add_argument("--short-max-val-samples", type=int, default=10000)
    parser.add_argument("--short-val-eval-samples", type=int, default=10000)
    parser.add_argument("--short-save-plot", action="store_true", default=True)
    parser.add_argument("--short-export-hdf5", action="store_true", help="Export HDF5 files for short sweep trials.")
    parser.add_argument("--skip-arch", action="store_true", help="Skip architecture sweep and use --architecture-for-scheduler.")
    parser.add_argument("--architecture-for-scheduler", type=parse_architectures, default=None, help="Single architecture to use when --skip-arch is set.")
    parser.add_argument("--skip-scheduler", action="store_true", help="Skip scheduler sweep.")
    parser.add_argument("--skip-long", action="store_true", help="Skip long confirmation runs.")
    parser.add_argument("--max-arch-trials", type=int, default=None)
    parser.add_argument("--max-scheduler-trials", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.short_epochs < 1 or args.long_epochs < 1:
        parser.error("epoch counts must be positive")
    if args.short_full_val_interval < 1 or args.long_full_val_interval < 1:
        parser.error("full-validation intervals must be positive")
    if args.skip_arch:
        if not args.architecture_for_scheduler or len(args.architecture_for_scheduler) != 1:
            parser.error("--skip-arch requires --architecture-for-scheduler with exactly one architecture")
    return args


def run_trial(
    *,
    args: argparse.Namespace,
    phase: str,
    label: str,
    params: dict[str, Any],
    output_dir: Path,
    log_path: Path,
    epochs: int,
    full_val_interval: int,
    export_hdf5: bool,
    save_plot: bool,
    use_short_caps: bool,
    resume: bool,
) -> dict[str, Any]:
    overrides = fixed_overrides(
        dataset=args.dataset.resolve(),
        checkpoint=args.ann_checkpoint.resolve(),
        output_dir=output_dir,
        label=label,
        hidden_dims=params["hidden_dims"],
        epochs=epochs,
        full_val_interval=full_val_interval,
        training_mode=params["training_mode"],
        num_sample_iter=params.get("num_sample_iter"),
        sample_period=params.get("sample_period"),
        crossover_epochs=params.get("crossover_epochs", []),
        export_hdf5=export_hdf5,
        save_plot=save_plot,
        device=args.device,
    )
    if use_short_caps:
        append_sample_caps(overrides, args)
    command = build_command(
        python=args.python,
        train_script=args.train_script.resolve(),
        config=args.config.resolve(),
        overrides=overrides,
    )
    return run_training(
        phase=phase,
        label=label,
        params=params,
        output_dir=output_dir,
        log_path=log_path,
        command=command,
        resume=resume,
    )


def main() -> None:
    args = parse_args()
    run_id = args.run_id or default_run_id()
    run_root = args.run_root.resolve() if args.run_root else (args.experiment_root / run_id).resolve()
    logs_dir = run_root / "logs"
    arch_root = run_root / "architecture_sweep"
    scheduler_root = run_root / "scheduler_sweep"
    long_root = run_root / "long_confirmations"
    resume = not args.no_resume

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not args.ann_checkpoint.exists():
        raise FileNotFoundError(f"ANN checkpoint not found: {args.ann_checkpoint}")

    arch_trials = [] if args.skip_arch else build_arch_trials(args)
    scheduler_trials = [] if args.skip_scheduler else build_scheduler_trials(args)
    plan = {
        "run_id": run_id,
        "run_root": run_root,
        "created_at": utc_now(),
        "config": args.config.resolve(),
        "train_script": args.train_script.resolve(),
        "dataset": args.dataset.resolve(),
        "ann_checkpoint": args.ann_checkpoint.resolve(),
        "fixed_snn_params": FIXED_SNN_PARAMS,
        "short_epochs": args.short_epochs,
        "long_epochs": args.long_epochs,
        "short_sample_caps": {
            "max_train_samples": args.short_max_train_samples,
            "max_val_samples": args.short_max_val_samples,
            "val_eval_samples": args.short_val_eval_samples,
        },
        "architecture_trial_count": len(arch_trials),
        "scheduler_trial_count": len(scheduler_trials),
        "architectures": args.architectures,
        "scheduler_trials": scheduler_trials,
        "resume": resume,
        "dry_run": args.dry_run,
    }
    write_json(run_root / "run_plan.json", plan)
    write_json(
        run_root / "status.json",
        {
            "updated_at": utc_now(),
            "status": "running",
            "run_root": run_root,
            "architecture_trials": len(arch_trials),
            "scheduler_trials": len(scheduler_trials),
        },
    )
    print(f"[plan] run_root={run_root}", flush=True)
    print(f"[plan] architecture_trials={len(arch_trials)} scheduler_trials={len(scheduler_trials)}", flush=True)
    print(f"[plan] short_epochs={args.short_epochs} long_epochs={args.long_epochs}", flush=True)

    if args.dry_run:
        print(f"[dry-run] wrote {run_root / 'run_plan.json'}", flush=True)
        return

    arch_results: list[dict[str, Any]] = []
    if args.skip_arch:
        selected_architecture = args.architecture_for_scheduler[0]
    else:
        for index, trial in enumerate(arch_trials, start=1):
            hidden_dims = trial["hidden_dims"]
            label = arch_label("arch", hidden_dims)
            trial_dir = arch_root / f"{index:03d}_{label}"
            log_path = logs_dir / f"{index:03d}_{label}.log"
            result = run_trial(
                args=args,
                phase="architecture_sweep",
                label=label,
                params=trial,
                output_dir=trial_dir,
                log_path=log_path,
                epochs=args.short_epochs,
                full_val_interval=args.short_full_val_interval,
                export_hdf5=args.short_export_hdf5,
                save_plot=args.short_save_plot,
                use_short_caps=True,
                resume=resume,
            )
            result["trial_index"] = index
            arch_results.append(result)
            ranked = write_phase_results(run_root, "architecture", arch_results)
            write_json(
                run_root / "status.json",
                {
                    "updated_at": utc_now(),
                    "status": "running_architecture_sweep",
                    "run_root": run_root,
                    "completed_architecture_trials": len(arch_results),
                    "architecture_trials": len(arch_trials),
                    "best_architecture_result": ranked[0] if ranked else None,
                },
            )
        ranked_arch = write_phase_results(run_root, "architecture", arch_results)
        if not ranked_arch:
            write_json(
                run_root / "status.json",
                {
                    "updated_at": utc_now(),
                    "status": "failed",
                    "reason": "No architecture trial completed with a full-validation MSE.",
                    "run_root": run_root,
                },
            )
            raise SystemExit("No successful architecture sweep results with full-validation MSE.")
        selected_architecture = ranked_arch[0]["params"]["hidden_dims"]
        write_json(
            run_root / "best_architecture.json",
            {
                "selected_at": utc_now(),
                "selection_metric": "best_full_val_snn_mse, then mean %, then median %",
                "hidden_dims": selected_architecture,
                "best_result": ranked_arch[0],
                "ranked_results": ranked_arch,
            },
        )
        print(f"[best-arch] h{format_arch(selected_architecture)}", flush=True)

    scheduler_results: list[dict[str, Any]] = []
    if not args.skip_scheduler:
        for index, trial in enumerate(scheduler_trials, start=1):
            params = {**trial, "hidden_dims": selected_architecture}
            label = scheduler_label("sched", trial)
            trial_dir = scheduler_root / f"{index:03d}_{label}"
            log_path = logs_dir / f"{index:03d}_{label}.log"
            result = run_trial(
                args=args,
                phase="scheduler_sweep",
                label=label,
                params=params,
                output_dir=trial_dir,
                log_path=log_path,
                epochs=args.short_epochs,
                full_val_interval=args.short_full_val_interval,
                export_hdf5=args.short_export_hdf5,
                save_plot=args.short_save_plot,
                use_short_caps=True,
                resume=resume,
            )
            result["trial_index"] = index
            scheduler_results.append(result)
            ranked = write_phase_results(run_root, "scheduler", scheduler_results)
            write_json(
                run_root / "status.json",
                {
                    "updated_at": utc_now(),
                    "status": "running_scheduler_sweep",
                    "run_root": run_root,
                    "completed_scheduler_trials": len(scheduler_results),
                    "scheduler_trials": len(scheduler_trials),
                    "best_scheduler_result": ranked[0] if ranked else None,
                },
            )
        ranked_scheduler = write_phase_results(run_root, "scheduler", scheduler_results)
        best_nonbaseline_scheduler = next(
            (result for result in ranked_scheduler if result.get("params", {}).get("training_mode") == "scheduler"),
            None,
        )
        write_json(
            run_root / "best_scheduler.json",
            {
                "selected_at": utc_now(),
                "selection_metric": "best_full_val_snn_mse, then mean %, then median %",
                "best_result_including_baseline": ranked_scheduler[0] if ranked_scheduler else None,
                "best_scheduler_result": best_nonbaseline_scheduler,
                "ranked_results": ranked_scheduler,
            },
        )
    else:
        ranked_scheduler = []
        best_nonbaseline_scheduler = None

    long_results: list[dict[str, Any]] = []
    if not args.skip_long:
        pure_params = {
            "phase": "long_architecture",
            "hidden_dims": selected_architecture,
            "training_mode": "pure_snn",
            "num_sample_iter": None,
            "sample_period": None,
            "crossover_epochs": [],
        }
        pure_label = arch_label(f"long_arch_e{args.long_epochs}", selected_architecture)
        long_results.append(
            run_trial(
                args=args,
                phase="long_pure_snn_confirmation",
                label=pure_label,
                params=pure_params,
                output_dir=long_root / pure_label,
                log_path=logs_dir / f"{pure_label}.log",
                epochs=args.long_epochs,
                full_val_interval=args.long_full_val_interval,
                export_hdf5=True,
                save_plot=True,
                use_short_caps=False,
                resume=resume,
            )
        )
        if best_nonbaseline_scheduler is not None:
            scheduler_params = {**best_nonbaseline_scheduler["params"], "phase": "long_scheduler"}
            scheduler_trial = best_nonbaseline_scheduler["params"]
            scheduler_long_label = scheduler_label(f"long_sched_e{args.long_epochs}", scheduler_trial)
            long_results.append(
                run_trial(
                    args=args,
                    phase="long_scheduler_confirmation",
                    label=scheduler_long_label,
                    params=scheduler_params,
                    output_dir=long_root / scheduler_long_label,
                    log_path=logs_dir / f"{scheduler_long_label}.log",
                    epochs=args.long_epochs,
                    full_val_interval=args.long_full_val_interval,
                    export_hdf5=True,
                    save_plot=True,
                    use_short_caps=False,
                    resume=resume,
                )
            )
        write_json(
            run_root / "long_confirmation_results.json",
            {
                "updated_at": utc_now(),
                "results": long_results,
                "ranked_results": successful_results(long_results),
            },
        )

    write_analysis(run_root, arch_results, scheduler_results, long_results)
    write_json(
        run_root / "status.json",
        {
            "updated_at": utc_now(),
            "status": "complete",
            "run_root": run_root,
            "selected_architecture": selected_architecture,
            "best_architecture_result": successful_results(arch_results)[0] if arch_results and successful_results(arch_results) else None,
            "best_scheduler_result": best_nonbaseline_scheduler,
            "long_results": long_results,
            "analysis_dir": run_root / "analysis",
        },
    )
    print(f"[complete] run_root={run_root}", flush=True)


if __name__ == "__main__":
    main()
