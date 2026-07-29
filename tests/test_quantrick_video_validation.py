import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


ROBOT_NAMES = [f"robot_{index:02d}" for index in range(16)]


def write_test_video(path: Path, robot_index: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64)
    )
    assert writer.isOpened()
    for frame_index in range(10):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :, robot_index % 3] = 40 + robot_index * 5
        x = 4 + frame_index * 7
        cv2.rectangle(frame, (x, 15), (x + 15, 45), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_video_validator_builds_complete_manifest_and_contact_sheets(tmp_path):
    video_dir = tmp_path / "videos"
    log_dir = tmp_path / "logs"
    contact_dir = tmp_path / "contacts"
    video_dir.mkdir()
    log_dir.mkdir()
    mapping_path = tmp_path / "video_mapping.tsv"
    baseline_path = tmp_path / "before_all_videos.txt"
    baseline_path.write_text("old_video.mp4\n")
    robots_config = tmp_path / "presets.yaml"
    robots_config.write_text(yaml.safe_dump({"record_robots": ROBOT_NAMES}))

    with mapping_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("robot_index", "robot_name", "video_path", "log_path"),
            delimiter="\t",
        )
        writer.writeheader()
        for index, robot_name in enumerate(ROBOT_NAMES):
            video_path = video_dir / f"new_{index:02d}.mp4"
            log_path = log_dir / f"{robot_name}.log"
            write_test_video(video_path, index)
            log_path.write_text("recording completed successfully\n")
            writer.writerow(
                {
                    "robot_index": index,
                    "robot_name": robot_name,
                    "video_path": video_path,
                    "log_path": log_path,
                }
            )

    output_path = tmp_path / "validation.json"
    subprocess.run(
        [
            sys.executable,
            "dev/codex/scripts/validate_quantrick_videos.py",
            "--mapping",
            str(mapping_path),
            "--baseline-snapshot",
            str(baseline_path),
            "--video-dir",
            str(video_dir),
            "--robots-config",
            str(robots_config),
            "--contact-dir",
            str(contact_dir),
            "--output",
            str(output_path),
            "--expected-duration",
            "1.0",
            "--duration-tolerance",
            "0.1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output_path.read_text())
    assert result["validated"] is True
    assert result["robot_count"] == 16
    assert all(item["validated"] for item in result["videos"])
    assert len(list(contact_dir.glob("*.jpg"))) == 16
