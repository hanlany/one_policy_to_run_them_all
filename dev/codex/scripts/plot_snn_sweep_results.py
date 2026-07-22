#!/usr/bin/env python3
"""Plot completed SNN sweep results for improvement_snn_v1 analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_RUN_ROOT = Path(
    "/app/one_policy_to_run_them_all/snn/experiments/snn_v1_points_1_2/20260619T050231Z"
)
EXPECTED_BEST_MSE = 0.011155231322348119
EXPECTED_LONG_MSE = 0.00592308546602726
EXPECTED_BEST_PARAMS = {
    "time_steps": 20,
    "input_weight": 1.0,
    "threshold": 0.2,
    "current_decay": 0.3,
    "voltage_decay": 0.02,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def first_existing_history(output_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    paths = sorted(output_dir.glob("*training_history.json"))
    if not paths:
        paths = sorted(output_dir.glob("*history*.json"))
    if not paths:
        return None, None
    path = paths[0]
    return path, load_json(path)


def non_none(values: list[Any]) -> list[tuple[int, float]]:
    result: list[tuple[int, float]] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append((index, number))
    return result


def value_at(history: dict[str, list[Any]], key: str, index: int | None) -> float | None:
    if index is None:
        return None
    values = history.get(key, [])
    if index >= len(values) or values[index] is None:
        return None
    return float(values[index])


def last_value(history: dict[str, list[Any]], key: str) -> float | None:
    values = non_none(history.get(key, []))
    return values[-1][1] if values else None


def summarize_history(history_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not history_payload:
        return {
            "history_best_mse": None,
            "history_best_epoch": None,
            "mean_pct_at_best": None,
            "median_pct_at_best": None,
            "final_full_val_mean_pct": None,
            "final_full_val_median_pct": None,
        }
    history = history_payload.get("history", history_payload)
    full_vals = non_none(history.get("full_val_snn_loss", []))
    best_index = None
    best_mse = None
    if full_vals:
        best_index, best_mse = min(full_vals, key=lambda item: item[1])
    return {
        "history_best_mse": best_mse,
        "history_best_epoch": None if best_index is None else best_index + 1,
        "mean_pct_at_best": value_at(history, "full_val_mean_percentage_error", best_index),
        "median_pct_at_best": value_at(history, "full_val_median_percentage_error", best_index),
        "final_full_val_mean_pct": last_value(history, "full_val_mean_percentage_error"),
        "final_full_val_median_pct": last_value(history, "full_val_median_percentage_error"),
    }


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}g}"


def row_label(row: dict[str, Any]) -> str:
    return (
        f"ts{int(row['time_steps'])} "
        f"iw{format_float(row['input_weight'], 3)} "
        f"thr{format_float(row['threshold'], 3)} "
        f"cd{format_float(row['current_decay'], 3)}"
    )


def load_rows(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    short = load_json(run_root / "short_sweep_results.json")
    best_params = load_json(run_root / "best_params.json")
    long_result = load_json(run_root / "long_best_result.json")
    rows: list[dict[str, Any]] = []
    for result in short["results"]:
        if result.get("exit_code") != 0 or result.get("best_full_val_snn_mse") is None:
            continue
        params = result["params"]
        output_dir = Path(result["output_dir"])
        history_path, history_payload = first_existing_history(output_dir)
        summary = summarize_history(history_payload)
        rows.append(
            {
                "trial_index": result.get("trial_index"),
                "label": result["label"],
                "short_label": row_label(params),
                "best_full_val_snn_mse": float(result["best_full_val_snn_mse"]),
                "result_best_epoch": result.get("best_epoch"),
                "duration_seconds": result.get("duration_seconds"),
                "output_dir": str(output_dir),
                "history_path": "" if history_path is None else str(history_path),
                "has_percentage_history": history_payload is not None
                and "full_val_mean_percentage_error" in history_payload.get("history", history_payload),
                **{key: params[key] for key in EXPECTED_BEST_PARAMS},
                **summary,
            }
        )
    rows.sort(key=lambda row: row["best_full_val_snn_mse"])
    return rows, short, best_params, long_result


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fields = [
        "rank",
        "trial_index",
        "label",
        "time_steps",
        "input_weight",
        "threshold",
        "current_decay",
        "voltage_decay",
        "best_full_val_snn_mse",
        "history_best_mse",
        "history_best_epoch",
        "mean_pct_at_best",
        "median_pct_at_best",
        "final_full_val_mean_pct",
        "final_full_val_median_pct",
        "duration_seconds",
        "has_percentage_history",
        "output_dir",
        "history_path",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **{field: row.get(field, "") for field in fields if field != "rank"}})


def plot_trial_ranking(rows: list[dict[str, Any]], best_label: str, top_k: int, output_path: Path) -> None:
    top = rows[:top_k]
    labels = [row["short_label"] for row in top]
    values = [row["best_full_val_snn_mse"] for row in top]
    colors = ["#1f77b4" if row["label"] != best_label else "#d62728" for row in top]
    fig, ax = plt.subplots(figsize=(max(11, top_k * 0.95), 5.5))
    bars = ax.bar(range(len(top)), values, color=colors)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Best full-val SNN MSE")
    ax.set_title(f"Top {top_k} SNN sweep trials by validation MSE")
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, top):
        pct = row.get("mean_pct_at_best") or row.get("final_full_val_mean_pct")
        med = row.get("median_pct_at_best") or row.get("final_full_val_median_pct")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.4f}\nmean {format_float(pct, 4)}%\nmed {format_float(med, 4)}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.margins(y=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(rows: list[dict[str, Any]], output_path: Path) -> None:
    time_steps = sorted({int(row["time_steps"]) for row in rows})
    current_decays = sorted({float(row["current_decay"]) for row in rows})
    input_weights = sorted({float(row["input_weight"]) for row in rows})
    thresholds = sorted({float(row["threshold"]) for row in rows})
    value_by_key = {
        (int(row["time_steps"]), float(row["current_decay"]), float(row["threshold"]), float(row["input_weight"])): row[
            "best_full_val_snn_mse"
        ]
        for row in rows
    }
    all_values = [row["best_full_val_snn_mse"] for row in rows]
    vmin, vmax = min(all_values), max(all_values)
    fig, axes = plt.subplots(
        len(time_steps),
        len(current_decays),
        figsize=(4.4 * len(current_decays), 3.6 * len(time_steps)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    last_image = None
    for r_index, ts in enumerate(time_steps):
        for c_index, cd in enumerate(current_decays):
            ax = axes[r_index][c_index]
            matrix = []
            for threshold in thresholds:
                matrix.append([value_by_key.get((ts, cd, threshold, iw), math.nan) for iw in input_weights])
            last_image = ax.imshow(matrix, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"time_steps={ts}, current_decay={cd:g}")
            ax.set_xticks(range(len(input_weights)))
            ax.set_xticklabels([f"{value:g}" for value in input_weights])
            ax.set_yticks(range(len(thresholds)))
            ax.set_yticklabels([f"{value:g}" for value in thresholds])
            ax.set_xlabel("input_weight")
            ax.set_ylabel("threshold")
            for y, threshold in enumerate(thresholds):
                for x, iw in enumerate(input_weights):
                    value = value_by_key.get((ts, cd, threshold, iw))
                    if value is not None:
                        ax.text(x, y, f"{value:.4f}", ha="center", va="center", fontsize=8, color="white")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), label="Best full-val SNN MSE", shrink=0.9)
    fig.suptitle("SNN sweep MSE heatmaps", y=0.995)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_time_steps_effect(rows: list[dict[str, Any]], output_path: Path) -> None:
    time_steps = sorted({int(row["time_steps"]) for row in rows})
    grouped = [[row["best_full_val_snn_mse"] for row in rows if int(row["time_steps"]) == ts] for ts in time_steps]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(grouped, labels=[str(ts) for ts in time_steps], showmeans=True)
    for index, values in enumerate(grouped, start=1):
        for item_index, value in enumerate(values):
            jitter = ((item_index % 7) - 3) * 0.018
            ax.scatter(index + jitter, value, color="#1f77b4", alpha=0.65, s=28)
    best_per_ts = [min(values) for values in grouped]
    ax.plot(range(1, len(time_steps) + 1), best_per_ts, color="#d62728", marker="o", label="best per time_steps")
    ax.set_xlabel("model.time_steps")
    ax.set_ylabel("Best full-val SNN MSE")
    ax.set_title("Effect of SNN time steps across swept dynamics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mse_vs_percentage(rows: list[dict[str, Any]], output_path: Path) -> None:
    time_steps = sorted({int(row["time_steps"]) for row in rows})
    current_decays = sorted({float(row["current_decay"]) for row in rows})
    colors = {ts: plt.cm.viridis(index / max(1, len(time_steps) - 1)) for index, ts in enumerate(time_steps)}
    markers = ["o", "s", "^", "D", "P", "X"]
    marker_by_cd = {cd: markers[index % len(markers)] for index, cd in enumerate(current_decays)}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for row in rows:
        y = row.get("mean_pct_at_best") or row.get("final_full_val_mean_pct")
        if y is None:
            continue
        ax.scatter(
            row["best_full_val_snn_mse"],
            y,
            color=colors[int(row["time_steps"])],
            marker=marker_by_cd[float(row["current_decay"])],
            edgecolor="black",
            linewidth=0.4,
            s=70,
            alpha=0.85,
        )
    ax.set_xlabel("Best full-val SNN MSE")
    ax.set_ylabel("Mean percentage error at best full-val checkpoint")
    ax.set_title("Does percentage error agree with MSE?")
    ax.grid(alpha=0.25)
    time_handles = [Line2D([0], [0], marker="o", color="w", label=f"ts={ts}", markerfacecolor=colors[ts], markersize=8) for ts in time_steps]
    decay_handles = [Line2D([0], [0], marker=marker_by_cd[cd], color="black", label=f"cd={cd:g}", linestyle="None", markersize=8) for cd in current_decays]
    first = ax.legend(handles=time_handles, title="time_steps", loc="upper left")
    ax.add_artist(first)
    ax.legend(handles=decay_handles, title="current_decay", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_short_training_curves(rows: list[dict[str, Any]], top_k: int, output_path: Path) -> None:
    top = rows[: min(5, top_k)]
    fig, (mse_ax, pct_ax) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for row in top:
        _, payload = first_existing_history(Path(row["output_dir"]))
        if not payload:
            continue
        history = payload.get("history", payload)
        epochs = list(range(1, len(history.get("train_snn_loss", [])) + 1))
        label = row["short_label"]
        mse_ax.plot(epochs, history.get("train_snn_loss", []), alpha=0.35, linestyle="--")
        mse_ax.plot(epochs, history.get("val_snn_loss", []), label=label)
        full_vals = non_none(history.get("full_val_snn_loss", []))
        if full_vals:
            mse_ax.scatter([i + 1 for i, _ in full_vals], [v for _, v in full_vals], s=28)
        pct_ax.plot(epochs, history.get("val_mean_percentage_error", []), label=f"{label} mean")
        pct_ax.plot(epochs, history.get("val_median_percentage_error", []), linestyle="--", alpha=0.75, label=f"{label} median")
    mse_ax.set_ylabel("SNN MSE")
    mse_ax.set_title(f"Training curves for top {len(top)} short-sweep trials")
    mse_ax.grid(alpha=0.25)
    mse_ax.legend(fontsize=8)
    pct_ax.set_xlabel("Epoch")
    pct_ax.set_ylabel("Percentage error")
    pct_ax.grid(alpha=0.25)
    pct_ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_long_training_curves(long_result: dict[str, Any], output_path: Path) -> None:
    _, payload = first_existing_history(Path(long_result["output_dir"]))
    if not payload:
        raise FileNotFoundError(f"No long training history JSON found in {long_result['output_dir']}")
    history = payload.get("history", payload)
    epochs = list(range(1, len(history.get("train_snn_loss", [])) + 1))
    fig, (mse_ax, pct_ax) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    mse_ax.plot(epochs, history.get("train_snn_loss", []), label="train MSE", alpha=0.75)
    mse_ax.plot(epochs, history.get("val_snn_loss", []), label="val subset MSE", alpha=0.85)
    full_vals = non_none(history.get("full_val_snn_loss", []))
    if full_vals:
        mse_ax.scatter([i + 1 for i, _ in full_vals], [v for _, v in full_vals], color="#d62728", s=28, label="full-val MSE")
    best_mse = long_result.get("best_full_val_snn_mse")
    if best_mse is not None:
        mse_ax.axhline(float(best_mse), color="#d62728", linestyle=":", label=f"best {float(best_mse):.6f}")
    mse_ax.set_ylabel("SNN MSE")
    mse_ax.set_title(f"Long best-parameter run: {long_result['label']}")
    mse_ax.grid(alpha=0.25)
    mse_ax.legend()
    pct_ax.plot(epochs, history.get("val_mean_percentage_error", []), label="val mean % error")
    pct_ax.plot(epochs, history.get("val_median_percentage_error", []), label="val median % error")
    full_mean = non_none(history.get("full_val_mean_percentage_error", []))
    full_median = non_none(history.get("full_val_median_percentage_error", []))
    if full_mean:
        pct_ax.scatter([i + 1 for i, _ in full_mean], [v for _, v in full_mean], s=24, label="full-val mean %")
    if full_median:
        pct_ax.scatter([i + 1 for i, _ in full_median], [v for _, v in full_median], marker="x", s=28, label="full-val median %")
    pct_ax.set_xlabel("Epoch")
    pct_ax.set_ylabel("Percentage error")
    pct_ax.grid(alpha=0.25)
    pct_ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def assert_expected_results(rows: list[dict[str, Any]], best_params: dict[str, Any], long_result: dict[str, Any]) -> None:
    if len(rows) != 36:
        raise AssertionError(f"Expected 36 successful short trials, found {len(rows)}")
    if not all(row["has_percentage_history"] for row in rows):
        missing = [row["label"] for row in rows if not row["has_percentage_history"]]
        raise AssertionError(f"Missing percentage-error history for {len(missing)} trials: {missing[:3]}")
    best = rows[0]
    if abs(best["best_full_val_snn_mse"] - EXPECTED_BEST_MSE) > 1e-12:
        raise AssertionError(f"Unexpected best short MSE: {best['best_full_val_snn_mse']}")
    for key, expected in EXPECTED_BEST_PARAMS.items():
        actual = best[key]
        if abs(float(actual) - float(expected)) > 1e-12:
            raise AssertionError(f"Unexpected best {key}: {actual} != {expected}")
        recorded = best_params["params"][key]
        if abs(float(recorded) - float(expected)) > 1e-12:
            raise AssertionError(f"best_params.json has unexpected {key}: {recorded} != {expected}")
    long_mse = float(long_result["best_full_val_snn_mse"])
    if abs(long_mse - EXPECTED_LONG_MSE) > 1e-12:
        raise AssertionError(f"Unexpected long-run best MSE: {long_mse}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SNN point-1/point-2 sweep results.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-assertions", action="store_true", help="Skip checks tied to the known completed run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or (run_root / "analysis_plots")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, short, best_params, long_result = load_rows(run_root)
    if not args.skip_assertions:
        assert_expected_results(rows, best_params, long_result)
    write_csv(rows, output_dir / "sweep_analysis_table.csv")
    plot_trial_ranking(rows, best_params["best_trial_label"], args.top_k, output_dir / "trial_ranking.png")
    plot_heatmaps(rows, output_dir / "sweep_heatmaps_mse.png")
    plot_time_steps_effect(rows, output_dir / "time_steps_effect.png")
    plot_mse_vs_percentage(rows, output_dir / "mse_vs_percentage_error.png")
    plot_short_training_curves(rows, args.top_k, output_dir / "top_trials_training_curves.png")
    plot_long_training_curves(long_result, output_dir / "long_best_training_curves.png")
    print(f"Loaded {len(rows)} successful short trials from {run_root}")
    print(f"Percentage-error histories: {sum(row['has_percentage_history'] for row in rows)}/{len(rows)}")
    print(f"Best short trial: {rows[0]['label']} mse={rows[0]['best_full_val_snn_mse']:.12f}")
    print(f"Long best mse: {float(long_result['best_full_val_snn_mse']):.12f}")
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
