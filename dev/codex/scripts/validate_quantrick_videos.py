#!/usr/bin/env python3
"""Validate QuanTrick robot videos and build per-robot contact sheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


FORBIDDEN_LOG_PATTERNS = {
    "traceback": re.compile(r"traceback", re.IGNORECASE),
    "nonfinite_action": re.compile(
        r"(?:\b(?:nan|inf)\b.{0,40}\baction\b|\baction\b.{0,40}\b(?:nan|inf)\b)",
        re.IGNORECASE,
    ),
    "checkpoint_mismatch": re.compile(r"checkpoint.{0,20}mismatch", re.IGNORECASE),
    "teacher_fallback": re.compile(
        r"(?:fallback|falling back).{0,30}teacher", re.IGNORECASE
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected_robots(config_path: Path) -> list[str]:
    payload = yaml.safe_load(config_path.read_text())
    robots = payload.get("record_robots")
    if not isinstance(robots, list) or not all(isinstance(item, str) for item in robots):
        raise ValueError(f"Invalid record_robots in {config_path}")
    return robots


def read_mapping(mapping_path: Path) -> list[dict[str, str]]:
    with mapping_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"robot_index", "robot_name", "video_path", "log_path"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"Invalid mapping columns in {mapping_path}")
    return rows


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        raise ValueError(f"Could not decode frame {index}")
    return frame


def build_contact_sheet(
    robot_name: str, frames: list[np.ndarray], output_path: Path
) -> None:
    target_width = 360
    panels = []
    labels = ("first", "middle", "final")
    for label, frame in zip(labels, frames):
        scale = target_width / frame.shape[1]
        resized = cv2.resize(
            frame,
            (target_width, max(1, round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        cv2.putText(
            resized,
            f"{robot_name}: {label}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        panels.append(resized)
    sheet = np.concatenate(panels, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise ValueError(f"Failed to write contact sheet {output_path}")


def validate_video(
    row: dict[str, str],
    video_dir: Path,
    baseline_names: set[str],
    contact_dir: Path,
    expected_duration: float,
    duration_tolerance: float,
) -> dict[str, Any]:
    video_path = Path(row["video_path"]).resolve()
    log_path = Path(row["log_path"]).resolve()
    errors: list[str] = []
    if video_path.parent != video_dir.resolve():
        errors.append("video is not directly under the configured video directory")
    if video_path.name in baseline_names:
        errors.append("video existed in the pre-recording snapshot")
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        errors.append("video is missing or empty")
    if not log_path.is_file():
        errors.append("per-robot recording log is missing")

    log_matches: list[str] = []
    if log_path.is_file():
        log_text = log_path.read_text(errors="replace")
        log_matches = [
            label
            for label, pattern in FORBIDDEN_LOG_PATTERNS.items()
            if pattern.search(log_text)
        ]
        if log_matches:
            errors.append("recording log contains forbidden diagnostics")

    metadata: dict[str, Any] = {}
    frames: list[np.ndarray] = []
    if video_path.is_file() and video_path.stat().st_size > 0:
        capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            errors.append("video does not open with OpenCV FFmpeg backend")
        else:
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            duration = frame_count / fps if fps > 0 else 0.0
            metadata = {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration,
            }
            if min(width, height, frame_count) <= 0 or fps <= 0:
                errors.append("video has non-positive frame metadata")
            if abs(duration - expected_duration) > duration_tolerance:
                errors.append(
                    f"duration {duration:.3f}s is outside "
                    f"{expected_duration:.3f}s +/- {duration_tolerance:.3f}s"
                )
            if frame_count > 0:
                try:
                    frames = [
                        read_frame(capture, index)
                        for index in (0, frame_count // 2, frame_count - 1)
                    ]
                except ValueError as exc:
                    errors.append(str(exc))
        capture.release()

    frame_stddev = [float(np.std(frame)) for frame in frames]
    if frames and any(value < 1.0 for value in frame_stddev):
        errors.append("first, middle, or final frame is blank")
    frame_pair_mae: list[float] = []
    if len(frames) == 3:
        frame_pair_mae = [
            float(np.mean(np.abs(frames[left].astype(np.float32) - frames[right].astype(np.float32))))
            for left, right in ((0, 1), (1, 2), (0, 2))
        ]
        if max(frame_pair_mae) < 0.5:
            errors.append("sampled frames appear frozen")

    contact_path = contact_dir / f"{int(row['robot_index']):02d}_{row['robot_name']}.jpg"
    if len(frames) == 3:
        build_contact_sheet(row["robot_name"], frames, contact_path)

    return {
        "robot_index": int(row["robot_index"]),
        "robot_name": row["robot_name"],
        "video_path": str(video_path),
        "video_sha256": sha256(video_path) if video_path.is_file() else None,
        "video_size_bytes": video_path.stat().st_size if video_path.is_file() else 0,
        "log_path": str(log_path),
        "log_sha256": sha256(log_path) if log_path.is_file() else None,
        "forbidden_log_matches": log_matches,
        "metadata": metadata,
        "sample_frame_stddev": frame_stddev,
        "sample_frame_pair_mae": frame_pair_mae,
        "contact_sheet_path": str(contact_path.resolve()) if len(frames) == 3 else None,
        "errors": errors,
        "validated": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--baseline-snapshot", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--robots-config", type=Path, required=True)
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-duration", type=float, default=10.0)
    parser.add_argument("--duration-tolerance", type=float, default=2.0)
    args = parser.parse_args()

    build_info = cv2.getBuildInformation()
    if not re.search(r"FFMPEG:\s+YES", build_info):
        raise RuntimeError("OpenCV was not built with FFmpeg support")

    expected_robots = load_expected_robots(args.robots_config)
    rows = read_mapping(args.mapping)
    mapped_names = [row["robot_name"] for row in rows]
    mapped_indices = [int(row["robot_index"]) for row in rows]
    if mapped_names != expected_robots or mapped_indices != list(range(len(expected_robots))):
        raise ValueError(
            "Mapping must contain every configured robot exactly once and in index order"
        )
    if len({row["video_path"] for row in rows}) != len(rows):
        raise ValueError("Mapping contains duplicate video paths")

    baseline_names = {
        line.strip()
        for line in args.baseline_snapshot.read_text().splitlines()
        if line.strip()
    }
    videos = [
        validate_video(
            row,
            args.video_dir,
            baseline_names,
            args.contact_dir,
            args.expected_duration,
            args.duration_tolerance,
        )
        for row in rows
    ]
    result = {
        "validated": len(videos) == 16 and all(item["validated"] for item in videos),
        "opencv_version": cv2.__version__,
        "opencv_ffmpeg_enabled": True,
        "expected_duration_seconds": args.expected_duration,
        "duration_tolerance_seconds": args.duration_tolerance,
        "robot_count": len(videos),
        "videos": videos,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["validated"] else 1)


if __name__ == "__main__":
    main()
