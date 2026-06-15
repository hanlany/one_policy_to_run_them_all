import os

import cv2
import mujoco


def sanitize_video_fragment(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def build_recording_path(output_dir, stage_name, env_id, robot_type, timestamp):
    safe_stage = sanitize_video_fragment(stage_name)
    safe_robot_type = sanitize_video_fragment(robot_type)
    file_name = f"{timestamp}_{safe_stage}_env{env_id:02d}_{safe_robot_type}.mp4"
    return os.path.join(output_dir, file_name)


def get_record_env_ids(record_robot_index, total_envs):
    if record_robot_index < 0:
        return list(range(total_envs))
    env_id = int(record_robot_index)
    if env_id < 0 or env_id >= total_envs:
        raise ValueError(f"record_robot_index {env_id} is out of range for {total_envs} envs.")
    return [env_id]


def get_record_timing(record_seconds, dt):
    dt = float(dt)
    if dt <= 0:
        raise ValueError(f"Invalid environment dt for recording: {dt}")
    fps = max(1, int(round(1.0 / dt)))
    total_frames = max(1, int(round(float(record_seconds) * fps)))
    return fps, total_frames


class VideoFrameWriter:
    def __init__(self, output_path, fps, frame_width=None, frame_height=None):
        self.output_path = output_path
        self.fps = float(max(1, int(round(fps))))
        self.frame_width = None if frame_width is None else int(frame_width)
        self.frame_height = None if frame_height is None else int(frame_height)
        self.writer = None
        if self.frame_width is not None and self.frame_height is not None:
            self._open_writer(self.frame_width, self.frame_height)

    def _open_writer(self, frame_width, frame_height):
        self.writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (int(frame_width), int(frame_height)),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.output_path}")
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)

    def write_rgb_frame(self, frame):
        if frame is None:
            raise RuntimeError("Cannot write an empty frame.")
        if self.writer is None:
            height, width = frame.shape[:2]
            self._open_writer(width, height)
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class OffscreenMujocoVideoRecorder:
    def __init__(self, model, output_path, fps, width=1280, height=720):
        self.output_path = output_path
        self.video_writer = None
        self.renderer = None
        self.camera = mujoco.MjvCamera()

        max_width = int(getattr(model.vis.global_, "offwidth", width))
        max_height = int(getattr(model.vis.global_, "offheight", height))
        safe_width = max(1, min(int(width), max_width))
        safe_height = max(1, min(int(height), max_height))

        # Keep the output inside MuJoCo's configured offscreen framebuffer limits.
        self.renderer = mujoco.Renderer(model, height=safe_height, width=safe_width)
        self.video_writer = VideoFrameWriter(
            output_path,
            fps=max(1, int(round(fps))),
            frame_width=safe_width,
            frame_height=safe_height,
        )
        self.set_follow_camera()

    def set_follow_camera(self):
        self.camera.fixedcamid = -1
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = 0
        self.camera.distance = 3.5
        self.camera.elevation = 0.0
        self.camera.azimuth = 90.0

    def capture(self, data):
        if self.renderer is None or self.video_writer is None:
            raise RuntimeError("Recorder is not initialized.")
        self.renderer.update_scene(data, camera=self.camera)
        self.video_writer.write_rgb_frame(self.renderer.render())

    def close(self):
        try:
            if self.video_writer is not None:
                self.video_writer.close()
        finally:
            close = getattr(self.renderer, "close", None)
            if callable(close):
                close()
