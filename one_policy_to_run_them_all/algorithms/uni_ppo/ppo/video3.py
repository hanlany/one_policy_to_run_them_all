import os
import inspect
import torch
import imageio
import numpy as np
import cv2

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.robots import anymal_b_simple, anymal_c_simple, base_robot, mini_cheetah, go1, aliengo, spot, spotmicro, siriusmid_belt, a1
from motion_imitation.real_a1 import a1_robot_real

import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

os.sys.path.insert(0, "/genloco-loihi/conversion")
import conversion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

robot_classes = {
    "laikago": base_robot.Base_robot,
    "A1": a1.A1,
    "ANYmal_B": anymal_b_simple.Anymal_b,
    "ANYmal": anymal_c_simple.Anymal_c,
    "Siriusmid_Belt": siriusmid_belt.Siriusmid_belt,
    "Mini Cheetah": mini_cheetah.Mini_cheetah,
    "Go1": go1.Go1,
    "Aliengo": aliengo.Aliengo,
    "Spot": spot.Spot,
    "SpotMicro": spotmicro.SpotMicro,
    "Real_A1": a1_robot_real.A1Robot,
}


def run_robot_comparison(robot_name, motion_file, steps=500):
    """Run ANN vs SNN comparison for one robot and return stacked frames."""

    print("Running comparison for robot:", robot_name)

    robot_class = robot_classes[robot_name]

    env_ann = env_builder.build_imitation_env(
        motion_files=[motion_file],
        num_parallel_envs=1,
        mode="test",
        enable_randomizer=False,
        enable_sync_root_rotation=False,
        enable_randomized_robot=False,
        enable_rendering=False,
        enable_phase_only=True,
        robot_class=robot_class,
        visualize=False,
    )

    env_snn = env_builder.build_imitation_env(
        motion_files=[motion_file],
        num_parallel_envs=1,
        mode="test",
        enable_randomizer=False,
        enable_rendering=False,
        enable_sync_root_rotation=False,
        enable_randomized_robot=False,
        enable_phase_only=True,
        robot_class=robot_class,
        visualize=False,
    )

    o_ann = env_ann.reset()
    o_snn = env_snn.reset()

    frames = []
    conversion.net.to(device)
    conversion.net.eval()

    with torch.no_grad():
        for stepnumber in range(steps):
            # print(".", end="", flush=True)
            # --- SNN ---
            inputs_snn = torch.tensor(o_snn).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
            snn_out = conversion.slayer_sdnn(inputs_snn)
            action_snn = snn_out.cpu().numpy().squeeze(0).squeeze(-1)
            o_snn, _, done_snn, _ = env_snn.step(action_snn)
            if done_snn:
                o_snn = env_snn.reset()

            # --- ANN ---
            inputs_ann = torch.tensor(o_ann).float().unsqueeze(0).to(device)
            ann_out = conversion.net(inputs_ann)
            action_ann = ann_out.cpu().numpy().squeeze(0)
            o_ann, _, done_ann, _ = env_ann.step(action_ann)
            if done_ann:
                o_ann = env_ann.reset()

            # Compute relative error
            relative_error_norm = conversion.get_error(conversion.net(inputs_snn[..., 0,0,0]).cpu().numpy().squeeze(0), action_snn)
            print(f"Relative Error in norm 2: {stepnumber} {relative_error_norm:5.4f}")

            # Render both envs
            frame_snn = env_snn.render(mode='rgb_array')
            frame_ann = env_ann.render(mode='rgb_array')

            # Convert to BGR for overlay
            frame_snn_bgr = cv2.cvtColor(frame_snn, cv2.COLOR_RGB2BGR)
            frame_ann_bgr = cv2.cvtColor(frame_ann, cv2.COLOR_RGB2BGR)

            h, w, _ = frame_snn_bgr.shape

            # === Overlays ===
            # Labels
        
            cv2.putText(frame_snn_bgr, f"SNN", (w - 80, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(frame_ann_bgr, f"ANN", (w - 80, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

            # Error bar on SNN frame
            bar_max_width = int(w * 0.4)
            bar_height = 20
            bar_x, bar_y = 20, h - 40

            error_percent = min(max(relative_error_norm * 100, 0), 100)
            bar_fill_width = int(bar_max_width * error_percent / 100.0)

            cv2.rectangle(frame_snn_bgr, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + bar_height), (180, 180, 180), -1)
            cv2.rectangle(frame_snn_bgr, (bar_x, bar_y), (bar_x + bar_fill_width, bar_y + bar_height), (0, 0, 255), -1)
            cv2.rectangle(frame_snn_bgr, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + bar_height), (0, 0, 0), 2)

            cv2.putText(frame_snn_bgr, f"err {error_percent:5.2f}%", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Convert back to RGB
            frame_snn = cv2.cvtColor(frame_snn_bgr, cv2.COLOR_BGR2RGB)
            frame_ann = cv2.cvtColor(frame_ann_bgr, cv2.COLOR_BGR2RGB)



            # Horizontal stack ANN vs SNN for this robot
            stacked = np.hstack((frame_snn, frame_ann))

            # Robot name centered at bottom
            (text_w, text_h), baseline = cv2.getTextSize(robot_name, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(stacked, robot_name, (stacked.shape[1] // 2 - text_w // 2, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

            frames.append(stacked)

    print("")
    return frames


def add_vertical_separator(frames, thickness=5, color=(0,0,0)): #(255, 255, 255)):
    """Insert a vertical line between frames in a row."""
    h, _, c = frames[0].shape
    sep = np.full((h, thickness, c), color, dtype=np.uint8)
    out = []
    for i, f in enumerate(frames):
        out.append(f)
        if i < len(frames) - 1:
            out.append(sep)
    return np.hstack(out)


def add_horizontal_separator(rows, thickness=5, color=(0,0,0)): #(255, 255, 255)):
    """Insert a horizontal line between rows."""
    _, w, c = rows[0].shape
    sep = np.full((thickness, w, c), color, dtype=np.uint8)
    out = []
    for i, r in enumerate(rows):
        out.append(r)
        if i < len(rows) - 1:
            out.append(sep)
    return np.vstack(out)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--robots", nargs="+", default=["A1", "Go1", "Mini Cheetah", "Spot", "Aliengo", "ANYmal"])
    arg_parser.add_argument("--motion_file", type=str, default="motion_imitation/data/motions/a1_pace.txt")
    arg_parser.add_argument("--steps", type=int, default=600)
    args = arg_parser.parse_args()

    all_robot_frames = []

    for robot in args.robots:
        frames = run_robot_comparison(robot, args.motion_file, steps=args.steps)
        all_robot_frames.append(frames)

    # Ensure all robots have the same number of frames (truncate to shortest)
    min_len = min(len(f) for f in all_robot_frames)
    all_robot_frames = [f[:min_len] for f in all_robot_frames]

    final_frames = []
    
    num_rows = 3
    num_cols = 2
    assert len(all_robot_frames) == num_rows * num_cols, "Need exactly 6 robots for a 3x2 grid."

    for i in range(min_len):
        row_frames = []
        for r in range(num_rows):
            start = r * num_cols
            end = start + num_cols
            row = add_vertical_separator([all_robot_frames[j][i] for j in range(start, end)])
            row_frames.append(row)
        grid_frame = add_horizontal_separator(row_frames)
        final_frames.append(grid_frame)


    print("Saving video to genloco_multi_robot.mp4")
    imageio.mimsave("genloco_multi_robot.mp4", final_frames, fps=20)


if __name__ == "__main__":
    main()
