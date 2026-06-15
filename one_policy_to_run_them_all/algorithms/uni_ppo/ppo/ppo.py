import math
import os
import psutil
import logging
import time
from collections import deque
import tree
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb
import torch
import cv2
import mujoco
import sys
sys.path.append("/app/one_policy_to_run_them_all/student")
try:
    from train_student import (
        DEFAULT_HIDDEN_DIMS,
        PolicyDistillationPipeline,
        SNNConversionConfig,
        StudentTrainingConfig,
        BootstrapTrainingConfig,
    )
except ImportError:
    print("Could not import student distillation utilities. Make sure train_student.py is in the path.")

from one_policy_to_run_them_all.algorithms.uni_ppo.ppo.general_properties import GeneralProperties
from one_policy_to_run_them_all.algorithms.uni_ppo.ppo.policy import get_policy
from one_policy_to_run_them_all.algorithms.uni_ppo.ppo.critic import get_critic

rlx_logger = logging.getLogger("rl_x")


class PolicyStageController:
    def __init__(self, teacher_predict, initial_stage="teacher"):
        self.teacher_predict = teacher_predict
        self.stage = initial_stage
        self.student_predict = None
        self.snn_predict = None

    def register_student(self, predict_fn):
        self.student_predict = predict_fn

    def register_snn(self, predict_fn):
        self.snn_predict = predict_fn

    def set_stage(self, stage):
        self.stage = stage

    def predict(self, policy_state, state, env_id=None, role="behavior"):
        if role == "teacher_label":
            return self.teacher_predict(policy_state, state, env_id)

        if self.stage == "snn" and self.snn_predict is not None:
            return self.snn_predict(state)
        if self.stage in {"student", "snn"} and self.student_predict is not None:
            return self.student_predict(state)
        return self.teacher_predict(policy_state, state, env_id)


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
        self.video_writer = VideoFrameWriter(output_path, fps=max(1, int(round(fps))), frame_width=safe_width, frame_height=safe_height)
        self.set_follow_camera(model)

    def set_follow_camera(self, model):
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


class PPO:
    def __init__(self, config, env, run_path, writer) -> None:
        self.config = config
        self.env = env.train_env
        self.eval_env = env.eval_env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.nr_eval_envs = config.environment.nr_eval_envs
        self.start_learning_rate = config.algorithm.start_learning_rate
        self.end_learning_rate = config.algorithm.end_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.softmax_temperature_min = config.algorithm.softmax_temperature_min
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.save_latest_frequency = config.algorithm.save_latest_frequency
        self.save_data = config.algorithm.save_data
        self.data_points = config.algorithm.data_points
        self.dagger_style = config.algorithm.dagger_style
        self.dagger_online = config.algorithm.dagger_online
        self.determine_fastest_cpu_for_gpu = config.algorithm.determine_fastest_cpu_for_gpu
        self.use_student = config.algorithm.use_student
        self.record = getattr(config.algorithm, "record", False)
        self.record_robot_index = int(getattr(config.algorithm, "record_robot_index", -1))
        self.record_seconds = float(getattr(config.algorithm, "record_seconds", 10.0))
        self.record_fps = int(getattr(config.algorithm, "record_fps", 60))
        self.record_dir = getattr(config.algorithm, "record_dir", "/app/one_policy_to_run_them_all/experiments/videos")
        self.multi_render = config.environment.multi_render
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        self.nr_rollouts = self.total_timesteps // (self.nr_steps * self.nr_envs)
        self.nr_env_types = len(config.environment.train_robot_types)
        self.nr_eval_env_types = len(config.environment.eval_robot_types)

        if self.evaluation_frequency % (self.nr_steps * self.nr_envs) != 0 and self.evaluation_frequency != -1:
            raise ValueError("Evaluation frequency must be a multiple of the number of steps and environments.")
        
        if self.save_latest_frequency % (self.nr_steps * self.nr_envs) != 0 and self.save_model:
            raise ValueError("Save latest frequency must be a multiple of the number of steps and environments.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        if self.determine_fastest_cpu_for_gpu and self.env.fastest_cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity([self.env.fastest_cpu_id,])
            rlx_logger.info(f"Using fastest CPU for GPU connection: {self.env.fastest_cpu_id}")
        
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.os_shape = self.env.single_observation_space.shape
        self.as_shape = self.env.single_action_space.shape
        
        self.policy, self.get_processed_action = get_policy(config, self.env)
        self.critic = get_critic(config, self.env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            learning_rate = self.end_learning_rate + fraction * (self.start_learning_rate - self.end_learning_rate)
            return learning_rate

        learning_rate = linear_schedule if self.start_learning_rate != self.end_learning_rate else self.start_learning_rate

        state = jnp.array([self.env.single_observation_space.sample() for _ in range(self.nr_envs)])

        # Create policy and critic state masks
        self.policy_general_state_mask = np.zeros(state.shape)
        self.critic_general_state_mask = np.zeros(state.shape)
        general_state_for_policy_names = [
            "trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity",
            "goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity",
            "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
            "p_gain", "d_gain", "action_scaling_factor",
            "mass",
            "robot_length", "robot_width", "robot_height"
        ]
        general_state_for_critic_names = [
            "trunk_x_velocity", "trunk_y_velocity", "trunk_z_velocity",
            "trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity",
            "goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity",
            "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
            "height_0",
            "p_gain", "d_gain", "action_scaling_factor",
            "mass",
            "robot_length", "robot_width", "robot_height"
        ]
        observation_name_to_ids = self.env.call("observation_name_to_id")
        for env_id in range(self.nr_envs):
            for name in general_state_for_policy_names:
                self.policy_general_state_mask[env_id, observation_name_to_ids[env_id][name]] = 1
            for name in general_state_for_critic_names:
                self.critic_general_state_mask[env_id, observation_name_to_ids[env_id][name]] = 1
        self.policy_general_state_mask = np.array(self.policy_general_state_mask, dtype=bool)
        self.critic_general_state_mask = np.array(self.critic_general_state_mask, dtype=bool)
        # eval
        if self.nr_eval_envs > 0:
            self.eval_policy_general_state_mask = np.zeros((self.nr_eval_envs, state.shape[-1]))
            eval_observation_name_to_ids = self.eval_env.call("observation_name_to_id")
            for env_id in range(self.nr_eval_envs):
                for name in general_state_for_policy_names:
                    self.eval_policy_general_state_mask[env_id, eval_observation_name_to_ids[env_id][name]] = 1
            self.eval_policy_general_state_mask = np.array(self.eval_policy_general_state_mask, dtype=bool)

        self.nr_envs_per_env_type = self.nr_envs // self.nr_env_types
        self.env_ids = np.array([self.nr_envs_per_env_type * i for i in range(self.nr_env_types + 1)])
        # eval
        if self.nr_eval_envs > 0:
            self.eval_nr_envs_per_env_type = self.nr_eval_envs // self.nr_eval_env_types
            self.eval_env_ids = np.array([self.eval_nr_envs_per_env_type * i for i in range(self.nr_eval_env_types + 1)])

        robot_types = self.env.call("robot_type")
        self.robot_types_list = np.array([robot_types[env_id] for env_id in self.env_ids[:-1]])
        # eval
        if self.nr_eval_envs > 0:
            eval_robot_types = self.eval_env.call("robot_type")
            self.eval_robot_types_list = np.array([eval_robot_types[env_id] for env_id in self.eval_env_ids[:-1]])

        self.nr_dynamic_joint_observations = np.array(self.env.call("nr_dynamic_joint_observations"))
        self.nr_dynamic_foot_observations = np.array(self.env.call("nr_dynamic_foot_observations"))
        self.dynamic_joint_observation_lengths = np.array(self.env.call("dynamic_joint_observation_length"))
        self.dynamic_foot_observation_lengths = np.array(self.env.call("dynamic_foot_observation_length"))
        self.single_dynamic_joint_observation_length = self.env.call("single_dynamic_joint_observation_length")[0]
        self.single_dynamic_foot_observation_length = self.env.call("single_dynamic_foot_observation_length")[0]
        self.dynamic_joint_description_size = self.env.call("dynamic_joint_description_size")[0]
        self.dynamic_foot_description_size = self.env.call("dynamic_foot_description_size")[0]
        # eval
        if self.nr_eval_envs > 0:
            self.eval_nr_dynamic_joint_observations = np.array(self.eval_env.call("nr_dynamic_joint_observations"))
            self.eval_nr_dynamic_foot_observations = np.array(self.eval_env.call("nr_dynamic_foot_observations"))
            self.eval_dynamic_joint_observation_lengths = np.array(self.eval_env.call("dynamic_joint_observation_length"))
            self.eval_dynamic_foot_observation_lengths = np.array(self.eval_env.call("dynamic_foot_observation_length"))
            self.eval_single_dynamic_joint_observation_length = self.eval_env.call("single_dynamic_joint_observation_length")[0]
            self.eval_single_dynamic_foot_observation_length = self.eval_env.call("single_dynamic_foot_observation_length")[0]
            self.eval_dynamic_joint_description_size = self.eval_env.call("dynamic_joint_description_size")[0]
            self.eval_dynamic_foot_description_size = self.eval_env.call("dynamic_foot_description_size")[0]

        self.missing_nr_of_actions = self.env.call("missing_nr_of_actions")
        self.all_missing_nr_of_actions = np.array([self.missing_nr_of_actions[env_id] for env_id in self.env_ids[:-1]])
        # eval
        if self.nr_eval_envs > 0:
            eval_missing_nr_of_actions = self.eval_env.call("missing_nr_of_actions")
            self.eval_all_missing_nr_of_actions = np.array([eval_missing_nr_of_actions[env_id] for env_id in self.eval_env_ids[:-1]])

        dummy_dynamic_joint_combined_state = state[:self.env_ids[1], :self.dynamic_joint_observation_lengths[self.env_ids[0]]].reshape((-1, self.nr_dynamic_joint_observations[self.env_ids[0]], self.single_dynamic_joint_observation_length))
        dummy_dynamic_joint_description = dummy_dynamic_joint_combined_state[:, :, :self.dynamic_joint_description_size]
        dummy_dynamic_joint_state = dummy_dynamic_joint_combined_state[:, :, self.dynamic_joint_description_size:]
        
        dummy_dynamic_foot_combined_state = state[:self.env_ids[1], self.dynamic_joint_observation_lengths[self.env_ids[0]]:self.dynamic_joint_observation_lengths[self.env_ids[0]] + self.dynamic_foot_observation_lengths[self.env_ids[0]]].reshape((-1, self.nr_dynamic_foot_observations[self.env_ids[0]], self.single_dynamic_foot_observation_length))
        dummy_dynamic_foot_description = dummy_dynamic_foot_combined_state[:, :, :self.dynamic_foot_description_size]
        dummy_dynamic_foot_state = dummy_dynamic_foot_combined_state[:, :, self.dynamic_foot_description_size:]

        dummy_general_policy_state = state[:self.env_ids[1], self.policy_general_state_mask[self.env_ids[0]]]
        dummy_general_critic_state = state[:self.env_ids[1], self.critic_general_state_mask[self.env_ids[0]]]

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_dynamic_joint_description, dummy_dynamic_joint_state, dummy_dynamic_foot_description, dummy_dynamic_foot_state, dummy_general_policy_state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, dummy_dynamic_joint_description, dummy_dynamic_joint_state, dummy_dynamic_foot_description, dummy_dynamic_foot_state, dummy_general_critic_state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_eps_track_perf_perc_average = 0.0
            self.best_model_file_name = "model_best_jax"
            self.latest_model_file_name = "model_latest_jax"
            best_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.best_model_file_name)
            latest_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.latest_model_file_name)
            self.best_model_checkpointer = orbax.checkpoint.Checkpointer(best_model_check_point_handler)
            self.latest_model_checkpointer = orbax.checkpoint.Checkpointer(latest_model_check_point_handler)

        if self.track_wandb and "SLURM_JOB_ID" in os.environ:
            rlx_logger.info(f"Slurm job id: {os.environ['SLURM_JOB_ID']}")
            wandb.config["SLURM_JOB_ID"] = os.environ["SLURM_JOB_ID"]

    
    def _sanitize_video_fragment(self, value):
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


    def _build_recording_path(self, output_dir, stage_name, env_id, robot_type, timestamp):
        safe_stage = self._sanitize_video_fragment(stage_name)
        safe_robot_type = self._sanitize_video_fragment(robot_type)
        file_name = f"{timestamp}_{safe_stage}_env{env_id:02d}_{safe_robot_type}.mp4"
        return os.path.join(output_dir, file_name)


    def _get_record_env_ids(self, total_envs):
        if self.record_robot_index < 0:
            return list(range(total_envs))
        env_id = int(self.record_robot_index)
        if env_id < 0 or env_id >= total_envs:
            raise ValueError(f"record_robot_index {env_id} is out of range for {total_envs} envs.")
        return [env_id]


    def _get_record_timing(self, dt):
        dt = float(dt)
        if dt <= 0:
            raise ValueError(f"Invalid environment dt for recording: {dt}")
        fps = max(1, int(round(1.0 / dt)))
        total_frames = max(1, int(round(self.record_seconds * fps)))
        return fps, total_frames


    def _record_vectorized_policy_videos(self, behavior_action_fn, stage_name):
        os.makedirs(self.record_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        models = self.env.call("model")
        dt_values = self.env.call("dt")
        robot_types = self.env.call("robot_type")
        output_paths = []
        record_env_ids = self._get_record_env_ids(len(models))

        for env_id in record_env_ids:
            model = models[env_id]
            dt = dt_values[env_id]
            robot_type = robot_types[env_id]
            record_fps, total_frames = self._get_record_timing(dt)
            output_path = self._build_recording_path(self.record_dir, stage_name, env_id, robot_type, timestamp)
            output_paths.append(output_path)
            recorder = OffscreenMujocoVideoRecorder(model, output_path, fps=record_fps)
            try:
                state, _ = self.env.reset()
                data_handles = self.env.call("data")
                recorder.capture(data_handles[env_id])
                for _ in range(total_frames - 1):
                    processed_action = behavior_action_fn(state)
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                    data_handles = self.env.call("data")
                    recorder.capture(data_handles[env_id])
            finally:
                recorder.close()

        return output_paths


    def _record_multi_render_policy_videos(self, behavior_action_fn, stage_name):
        os.makedirs(self.record_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        render_wrapper = getattr(self.env, "env", None)
        viewer = getattr(render_wrapper, "viewer", None)
        if viewer is None:
            raise RuntimeError("Multi-render recording requires the MultiRenderWrapper viewer.")

        base_env = self.env.unwrapped
        models = base_env.call("model")
        dt_values = base_env.call("dt")
        robot_types = base_env.call("robot_type")
        output_paths = []
        record_env_ids = self._get_record_env_ids(len(models))
        original_active_env_id = getattr(base_env, "active_env_id", 0)
        original_env_to_render = getattr(render_wrapper, "env_to_render", original_active_env_id)

        try:
            for env_id in record_env_ids:
                model = models[env_id]
                dt = dt_values[env_id]
                robot_type = robot_types[env_id]
                record_fps, total_frames = self._get_record_timing(dt)
                output_path = self._build_recording_path(self.record_dir, stage_name, env_id, robot_type, timestamp)
                output_paths.append(output_path)
                writer = VideoFrameWriter(output_path, fps=record_fps)
                try:
                    base_env.active_env_id = env_id
                    render_wrapper.env_to_render = env_id
                    viewer.load_new_model(model)
                    state, _ = base_env.reset()
                    writer.write_rgb_frame(viewer.render_to_rgb(base_env.call("data")[env_id]))
                    for _ in range(total_frames - 1):
                        processed_action = behavior_action_fn(state, env_id_override=env_id)
                        state, reward, terminated, truncated, info = base_env.step(jax.device_get(processed_action))
                        writer.write_rgb_frame(viewer.render_to_rgb(base_env.call("data")[env_id]))
                finally:
                    writer.close()
        finally:
            base_env.active_env_id = original_active_env_id
            render_wrapper.env_to_render = original_env_to_render
            if models:
                viewer.load_new_model(models[original_active_env_id])

        return output_paths


    def _record_policy_videos(self, behavior_action_fn, stage_name):
        if self.multi_render:
            return self._record_multi_render_policy_videos(behavior_action_fn, stage_name)
        return self._record_vectorized_policy_videos(behavior_action_fn, stage_name)

    def train(self):
        @jax.jit
        def train_loop():
            @jax.jit
            def get_action_and_value(policy_state, critic_state, state, key):
                @partial(jax.jit, static_argnums=(1,2))
                def compute_single_action_and_value(key, env_id_0, env_id_1):
                    state_b = state[env_id_0:env_id_1]

                    dynamic_joint_combined_state = state_b[:, :self.dynamic_joint_observation_lengths[env_id_0]].reshape((-1, self.nr_dynamic_joint_observations[env_id_0], self.single_dynamic_joint_observation_length))
                    dynamic_joint_description = dynamic_joint_combined_state[:, :, :self.dynamic_joint_description_size]
                    dynamic_joint_state = dynamic_joint_combined_state[:, :, self.dynamic_joint_description_size:]

                    dynamic_foot_combined_state = state_b[:, self.dynamic_joint_observation_lengths[env_id_0]:self.dynamic_joint_observation_lengths[env_id_0] + self.dynamic_foot_observation_lengths[env_id_0]].reshape((-1, self.nr_dynamic_foot_observations[env_id_0], self.single_dynamic_foot_observation_length))
                    dynamic_foot_description = dynamic_foot_combined_state[:, :, :self.dynamic_foot_description_size]
                    dynamic_foot_state = dynamic_foot_combined_state[:, :, self.dynamic_foot_description_size:]

                    action_mean, action_logstd = self.policy.apply(policy_state.params, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, state_b[:, self.policy_general_state_mask[env_id_0]])
                    action_std = jnp.exp(action_logstd)

                    key, subkey = jax.random.split(key)
                    action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                    log_prob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                    action = jnp.concatenate([action, jnp.zeros((self.nr_envs_per_env_type, self.missing_nr_of_actions[env_id_0]))], axis=1)
                    processed_action = self.get_processed_action(action)

                    value = self.critic.apply(critic_state.params, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, state_b[:, self.critic_general_state_mask[env_id_0]]).squeeze(-1)

                    return processed_action, action, value, log_prob.sum(1), key
            
                processed_actions = []
                actions = []
                values = []
                log_probs = []

                for i in range(self.env_ids.shape[0] - 1):
                    env_id_0 = self.env_ids[i]
                    env_id_1 = self.env_ids[i + 1]
                    processed_action, action, value, log_prob_sum, key = compute_single_action_and_value(key, env_id_0, env_id_1)
                    processed_actions.append(processed_action)
                    actions.append(action)
                    values.append(value)
                    log_probs.append(log_prob_sum)
                
                return jnp.concatenate(processed_actions), jnp.concatenate(actions), jnp.concatenate(values), jnp.concatenate(log_probs), key


            @jax.jit
            def calculate_gae_advantages(critic_state, next_states, rewards, terminations, values):
                @partial(jax.jit, static_argnums=(0,1))
                def get_next_value(env_id_0, env_id_1):
                    next_states_b = next_states[:, env_id_0:env_id_1]

                    dynamic_joint_combined_next_state = next_states_b[:, :, :self.dynamic_joint_observation_lengths[env_id_0]].reshape((self.nr_steps, self.nr_envs_per_env_type, self.nr_dynamic_joint_observations[env_id_0], self.single_dynamic_joint_observation_length))
                    dynamic_joint_description = dynamic_joint_combined_next_state[:, :, :, :self.dynamic_joint_description_size]
                    dynamic_joint_state = dynamic_joint_combined_next_state[:, :, :, self.dynamic_joint_description_size:]

                    dynamic_foot_combined_next_state = next_states_b[:, :, self.dynamic_joint_observation_lengths[env_id_0]:self.dynamic_joint_observation_lengths[env_id_0] + self.dynamic_foot_observation_lengths[env_id_0]].reshape((self.nr_steps, self.nr_envs_per_env_type, self.nr_dynamic_foot_observations[env_id_0], self.single_dynamic_foot_observation_length))
                    dynamic_foot_description = dynamic_foot_combined_next_state[:, :, :, :self.dynamic_foot_description_size]
                    dynamic_foot_state = dynamic_foot_combined_next_state[:, :, :, self.dynamic_foot_description_size:]

                    next_value = self.critic.apply(critic_state.params, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, next_states_b[:, :, self.critic_general_state_mask[env_id_0]]).squeeze(-1)

                    return next_value
            
                def compute_advantages(carry, t):
                    prev_advantage, delta, terminations = carry
                    advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                    return (advantage, delta, terminations), advantage

                next_values = []
                for i in range(self.env_ids.shape[0] - 1):
                    env_id_0 = self.env_ids[i]
                    env_id_1 = self.env_ids[i + 1]
                    next_value = get_next_value(env_id_0, env_id_1)
                    next_values.append(next_value)
                next_values = jnp.concatenate(next_values, axis=1)

                delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                init_advantages = delta[-1]
                _, advantages = jax.lax.scan(compute_advantages, (init_advantages, delta, terminations), jnp.arange(self.nr_steps - 2, -1, -1))
                advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                returns = advantages + values

                return advantages, returns
            

            @jax.jit
            def update(policy_state, critic_state, states, actions, advantages, returns, values, log_probs, key):
                def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b):
                    @partial(jax.jit, static_argnums=(5))
                    def single_loss_fn(state, action, log_prob, return_, advantage, env_type_id):
                        state_b = state[env_type_id]
                        action_b = action[env_type_id, :self.as_shape[0] - self.all_missing_nr_of_actions[env_type_id]]
                        log_prob_b = log_prob[env_type_id]
                        return_b = return_[env_type_id]
                        advantage_b = advantage[env_type_id]

                        dynamic_joint_combined_state = state_b[:self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type]].reshape((self.nr_dynamic_joint_observations[env_type_id*self.nr_envs_per_env_type], self.single_dynamic_joint_observation_length))
                        dynamic_joint_description = dynamic_joint_combined_state[:, :self.dynamic_joint_description_size]
                        dynamic_joint_state = dynamic_joint_combined_state[:, self.dynamic_joint_description_size:]

                        dynamic_foot_combined_state = state_b[self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type]:self.dynamic_joint_observation_lengths[env_type_id*self.nr_envs_per_env_type] + self.dynamic_foot_observation_lengths[env_type_id*self.nr_envs_per_env_type]].reshape((self.nr_dynamic_foot_observations[env_type_id*self.nr_envs_per_env_type], self.single_dynamic_foot_observation_length))
                        dynamic_foot_description = dynamic_foot_combined_state[:, :self.dynamic_foot_description_size]
                        dynamic_foot_state = dynamic_foot_combined_state[:, self.dynamic_foot_description_size:]

                        # Policy loss
                        action_mean, action_logstd = self.policy.apply(policy_params, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, state_b[self.policy_general_state_mask[env_type_id*self.nr_envs_per_env_type]])
                        action_std = jnp.exp(action_logstd)
                        action_std_mean = action_std.mean()
                        new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                        new_log_prob = new_log_prob.sum()
                        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)

                        logratio = new_log_prob - log_prob_b
                        ratio = jnp.exp(logratio)
                        approx_kl_div = (ratio - 1) - logratio
                        clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                        pg_loss1 = -advantage_b * ratio
                        pg_loss2 = -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        pg_loss = jnp.maximum(pg_loss1, pg_loss2)

                        entropy_loss = entropy.sum()

                        # Critic loss
                        new_value = self.critic.apply(critic_params, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, state_b[self.critic_general_state_mask[env_type_id*self.nr_envs_per_env_type]]).squeeze(-1)
                        critic_loss = 0.5 * (new_value - return_b) ** 2

                        # Combine losses
                        loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss

                        # Create metrics
                        metrics = {
                            "loss/policy_gradient_loss": pg_loss,
                            "loss/critic_loss": critic_loss,
                            "loss/entropy_loss": entropy_loss,
                            "policy_ratio/approx_kl": approx_kl_div,
                            "policy_ratio/clip_fraction": clip_fraction,
                            "policy/std_dev": action_std_mean
                        }

                        return loss, metrics
                
                    losses = []
                    metrics = []
                    for env_type_id in range(self.nr_env_types):
                        loss, metric = single_loss_fn(state_b, action_b, log_prob_b, return_b, advantage_b, env_type_id)
                        losses.append(loss)
                        metrics.append(metric)
                    loss_mean = jnp.mean(jnp.array(losses))
                    metrics_mean = jax.tree_map(lambda *x: jnp.mean(jnp.array(x)), *metrics)
                    
                    return loss_mean, (metrics_mean)


                batch_states = states.reshape((self.nr_steps, self.nr_env_types, self.nr_envs_per_env_type, *self.os_shape)).transpose((0, 2, 1, 3)).reshape((self.nr_steps * self.nr_envs_per_env_type, self.nr_env_types, *self.os_shape))
                batch_actions = actions.reshape((self.nr_steps, self.nr_env_types, self.nr_envs_per_env_type, *self.as_shape)).transpose((0, 2, 1, 3)).reshape((self.nr_steps * self.nr_envs_per_env_type, self.nr_env_types, *self.as_shape))
                batch_log_probs = log_probs.reshape((self.nr_steps, self.nr_env_types, self.nr_envs_per_env_type)).transpose((0, 2, 1)).reshape((self.nr_steps * self.nr_envs_per_env_type, self.nr_env_types))
                batch_returns = returns.reshape((self.nr_steps, self.nr_env_types, self.nr_envs_per_env_type)).transpose((0, 2, 1)).reshape((self.nr_steps * self.nr_envs_per_env_type, self.nr_env_types))
                batch_advantages = advantages.reshape((self.nr_steps, self.nr_env_types, self.nr_envs_per_env_type)).transpose((0, 2, 1)).reshape((self.nr_steps * self.nr_envs_per_env_type, self.nr_env_types))

                vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
                safe_mean = lambda x: jnp.mean(x) if x is not None else x
                mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

                key, subkey = jax.random.split(key)
                batch_indices = jnp.tile(jnp.arange(self.nr_steps * self.nr_envs_per_env_type), (self.nr_epochs, 1))
                batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size // self.nr_env_types))

                def minibatch_update(carry, minibatch_indices):
                    policy_state, critic_state = carry

                    minibatch_advantages = batch_advantages[minibatch_indices]
                    minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages, axis=0)) / (jnp.std(minibatch_advantages, axis=0) + 1e-8)

                    (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                        policy_state.params,
                        critic_state.params,
                        batch_states[minibatch_indices],
                        batch_actions[minibatch_indices],
                        batch_log_probs[minibatch_indices],
                        batch_returns[minibatch_indices],
                        minibatch_advantages,
                    )

                    policy_state = policy_state.apply_gradients(grads=policy_gradients)
                    critic_state = critic_state.apply_gradients(grads=critic_gradients)

                    metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                    metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                    carry = policy_state, critic_state

                    return carry, (metrics)
                
                init_carry = policy_state, critic_state
                carry, (metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                policy_state, critic_state = carry

                # Calculate mean metrics
                mean_metrics = {key: jnp.mean(metrics[key]) for key in metrics}
                mean_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                mean_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                mean_metrics["policy/joint_softmax_temp"] = jnp.exp(policy_state.params["params"]["joint_log_softmax_temperature"])[0] + self.softmax_temperature_min
                mean_metrics["policy/foot_softmax_temp"] = jnp.exp(policy_state.params["params"]["foot_log_softmax_temperature"])[0] + self.softmax_temperature_min
                mean_metrics["critic/joint_softmax_temp"] = jnp.exp(critic_state.params["params"]["joint_log_softmax_temperature"])[0] + self.softmax_temperature_min
                mean_metrics["critic/foot_softmax_temp"] = jnp.exp(critic_state.params["params"]["foot_log_softmax_temperature"])[0] + self.softmax_temperature_min

                return policy_state, critic_state, mean_metrics, key
            

            @partial(jax.jit, backend="cpu")
            def get_eval_action(policy_state, state, key):
                @partial(jax.jit, static_argnums=(1,2))
                def get_single_deterministic_eval_action(key, env_id_0, env_id_1):
                    state_b = state[env_id_0:env_id_1]

                    dynamic_joint_combined_state = state_b[:, :self.eval_dynamic_joint_observation_lengths[env_id_0]].reshape((-1, self.eval_nr_dynamic_joint_observations[env_id_0], self.eval_single_dynamic_joint_observation_length))
                    dynamic_joint_description = dynamic_joint_combined_state[:, :, :self.eval_dynamic_joint_description_size]
                    dynamic_joint_state = dynamic_joint_combined_state[:, :, self.eval_dynamic_joint_description_size:]

                    dynamic_foot_combined_state = state_b[:, self.eval_dynamic_joint_observation_lengths[env_id_0]:self.eval_dynamic_joint_observation_lengths[env_id_0] + self.eval_dynamic_foot_observation_lengths[env_id_0]].reshape((-1, self.eval_nr_dynamic_foot_observations[env_id_0], self.eval_single_dynamic_foot_observation_length))
                    dynamic_foot_description = dynamic_foot_combined_state[:, :, :self.eval_dynamic_foot_description_size]
                    dynamic_foot_state = dynamic_foot_combined_state[:, :, self.eval_dynamic_foot_description_size:]

                    action_mean, action_logstd = self.policy.apply(policy_state.params, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, state_b[:, self.eval_policy_general_state_mask[env_id_0]])
                    action_std = jnp.exp(action_logstd)

                    key, subkey = jax.random.split(key)
                    action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                    action = jnp.concatenate([action, jnp.zeros((self.eval_nr_envs_per_env_type, self.eval_all_missing_nr_of_actions[env_id_0]))], axis=1)
                    processed_action = self.get_processed_action(action)

                    return processed_action, key
            
                processed_actions = []
                for i in range(self.eval_env_ids.shape[0] - 1):
                    env_id_0 = self.eval_env_ids[i]
                    env_id_1 = self.eval_env_ids[i + 1]
                    processed_action, key = get_single_deterministic_eval_action(key, env_id_0, env_id_1)
                    processed_actions.append(processed_action)
                
                return jnp.concatenate(processed_actions), key
            

            def env_reset_callback():
                state, _ = self.env.reset()
                state = state.astype(np.float32)
                return state
            

            def reset_step_info_collection_callback():
                self.step_info_collection = {}


            def env_step_callback(action):
                next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(action))
                next_state = next_state.astype(np.float32)
                reward = reward.astype(np.float32)

                actual_next_state = next_state.copy()
                done = terminated | truncated
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = self.env.get_final_observation_at_index(info, i)
                        for j, env_id in enumerate(self.env_ids):
                            if i < env_id:
                                self.saving_return_buffer.append(self.env.get_final_info_value_at_index(info, f"episode_return_{self.robot_types_list[j-1]}", i))
                                break
                self.current_nr_episodes += np.sum(done)

                for key, info_value in self.env.get_logging_info_dict(info).items():
                    self.step_info_collection.setdefault(key, []).extend(info_value)

                return next_state, actual_next_state, reward, terminated, truncated
            

            def evaluate_callback(policy_state, old_state, key):
                self.evaluation_metrics = {}
                if self.nr_eval_envs > 0 and self.global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                    self.set_eval_mode()
                    eval_state, _ = self.eval_env.reset()
                    eval_nr_episodes = np.zeros(self.nr_eval_env_types, dtype=int)
                    for robot_type in self.eval_robot_types_list:
                        self.evaluation_metrics[f"eval/episode_return/{robot_type}"] = []
                        self.evaluation_metrics[f"eval/episode_length/{robot_type}"] = []
                        self.evaluation_metrics[f"eval/track_perf_perc/{robot_type}"] = []
                        self.evaluation_metrics[f"eval/eps_track_perf_perc/{robot_type}"] = []
                    while True:
                        eval_processed_action, key = get_eval_action(policy_state, eval_state, key)
                        eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(jax.device_get(eval_processed_action))
                        for eval_key, eval_info_value in self.eval_env.get_logging_info_dict(eval_info).items():
                            if "track_perf_perc" in eval_key:
                                name = eval_key.replace("env_info", "eval")
                                self.evaluation_metrics[name].extend(eval_info_value)
                        eval_done = eval_terminated | eval_truncated
                        for i, eval_single_done in enumerate(eval_done):
                            if eval_single_done:
                                for j, eval_env_id in enumerate(self.eval_env_ids):
                                    if i < eval_env_id and eval_nr_episodes[i] < self.evaluation_episodes:
                                        eval_nr_episodes[i] += 1  # works because only one env per type
                                        self.evaluation_metrics[f"eval/episode_return/{self.eval_robot_types_list[j-1]}"].append(
                                            self.eval_env.get_final_info_value_at_index(eval_info, f"episode_return_{self.eval_robot_types_list[j-1]}", i)
                                        )
                                        self.evaluation_metrics[f"eval/episode_length/{self.eval_robot_types_list[j-1]}"].append(
                                            self.eval_env.get_final_info_value_at_index(eval_info, f"episode_length_{self.eval_robot_types_list[j-1]}", i)
                                        )
                                        self.evaluation_metrics[f"eval/track_perf_perc/{self.eval_robot_types_list[j-1]}"].append(
                                            self.eval_env.get_final_info_value_at_index(eval_info, f"env_info/track_perf_perc/{self.eval_robot_types_list[j-1]}", i)
                                        )
                                        break
                        if np.all(eval_nr_episodes == self.evaluation_episodes):
                            break
                    self.evaluation_metrics = {key: np.mean(value) for key, value in self.evaluation_metrics.items()}
                    state, _ = self.env.reset()
                    self.set_train_mode()
                    return state.astype(np.float32), key
                else:
                    return old_state, key


            def save_and_log_callback(policy_state, critic_state, optimization_metrics):    
                sum_eps_track_perf_perc = 0
                sum_episode_return = 0
                sum_episode_length = 0
                count_eps_track_perf_perc = 0
                count_episode_return = 0
                count_episode_length = 0
                for key, value in self.step_info_collection.items():
                    self.step_info_collection[key] = np.mean(value)
                    if "eps_track_perf_perc" in key:
                        sum_eps_track_perf_perc += self.step_info_collection[key]
                        count_eps_track_perf_perc += 1
                    if "episode_return" in key:
                        sum_episode_return += self.step_info_collection[key]
                        count_episode_return += 1
                    if "episode_length" in key:
                        sum_episode_length += self.step_info_collection[key]
                        count_episode_length += 1
                if count_eps_track_perf_perc > 0:
                    self.step_info_collection["env_info/eps_track_perf_perc/average"] = sum_eps_track_perf_perc / count_eps_track_perf_perc
                if count_episode_return > 0:
                    self.step_info_collection["rollout/episode_return/average"] = sum_episode_return / count_episode_return
                if count_episode_length > 0:
                    self.step_info_collection["rollout/episode_length/average"] = sum_episode_length / count_episode_length

                # Saving
                if self.save_model:
                    eps_track_perf_perc_average = self.step_info_collection["env_info/eps_track_perf_perc/average"]
                    if eps_track_perf_perc_average > self.best_eps_track_perf_perc_average:
                        self.best_eps_track_perf_perc_average = eps_track_perf_perc_average
                        self.save(policy_state, critic_state, "best")
                if self.save_model and self.global_step % self.save_latest_frequency == 0:
                    self.save(policy_state, critic_state, "latest")

                # Logging
                current_time = time.time()
                sps = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time))
                self.last_time = current_time
                time_metrics = {
                    "time/sps": sps
                }

                self.global_step += self.nr_steps * self.nr_envs
                self.current_nr_updates += self.nr_epochs * self.nr_minibatches
                steps_metrics = {
                    "steps/nr_env_steps": self.global_step,
                    "steps/nr_env_type_steps": self.global_step // self.nr_env_types,
                    "steps/nr_updates": self.current_nr_updates,
                    "steps/nr_episodes": self.current_nr_episodes
                }

                rollout_info_metrics = {}
                env_info_metrics = {}
                info_names = list(self.step_info_collection.keys())
                for info_name in info_names:
                    log_info_name = info_name
                    metric_group = ""
                    for robot_type in self.robot_types_list:
                        if info_name in [f"episode_return_{robot_type}", f"episode_length_{robot_type}"]:
                            log_info_name = robot_type
                            metric_group = f"rollout/{info_name.replace(f'_{robot_type}', '')}/"
                            break
                    metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                    value = self.step_info_collection[info_name]
                    if value == value:  # Check if value is NaN
                        metric_dict[f"{metric_group}{log_info_name}"] = value
                
                additional_metrics = {**rollout_info_metrics, **self.evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics}

                self.start_logging(self.global_step)
                for key, value in additional_metrics.items():
                    self.log(key, value, self.global_step)
                for key, value in optimization_metrics.items():
                    self.log(key, value, self.global_step)
                self.end_logging()


            self.set_train_mode()

            state_shape = jax.ShapeDtypeStruct(self.np_state_shape, jnp.float32)
            reward_shape = jax.ShapeDtypeStruct((self.nr_envs,), jnp.float32)
            terminated_shape = jax.ShapeDtypeStruct((self.nr_envs,), jnp.bool_)
            truncated_shape = jax.ShapeDtypeStruct((self.nr_envs,), jnp.bool_)
            combined_callback_shapes = (state_shape, state_shape, reward_shape, terminated_shape, truncated_shape)
            key_shape = jax.ShapeDtypeStruct(self.key.shape, self.key.dtype)
            state_key_shape = (state_shape, key_shape)

            state = jax.pure_callback(env_reset_callback, state_shape)

            def train(carry, _):
                def rollout(carry, _):
                    policy_state, critic_state, state, key = carry
                    processed_action, action, value, log_prob, key = get_action_and_value(policy_state, critic_state, state, key)
                    next_state, actual_next_state, reward, terminated, truncated = jax.pure_callback(env_step_callback, combined_callback_shapes, processed_action)

                    batch = (state, actual_next_state, action, reward, value, terminated, log_prob)

                    return (policy_state, critic_state, next_state, key), batch


                # Acting
                jax.debug.callback(reset_step_info_collection_callback)
                new_carry, batch = jax.lax.scan(rollout, carry, jnp.arange(self.nr_steps))


                # Calculating advantages and returns
                policy_state, critic_state, state, key = new_carry
                states, actual_next_states, actions, rewards, values, terminations, log_probs = batch
                advantages, returns = calculate_gae_advantages(critic_state, actual_next_states, rewards, terminations, values)


                # Optimizing
                policy_state, critic_state, optimization_metrics, key = update(
                    policy_state, critic_state,
                    states, actions, advantages, returns, values, log_probs,
                    key
                )


                # Evaluating
                state, key = jax.pure_callback(evaluate_callback, state_key_shape, policy_state, state, key)


                # Saving and logging
                jax.debug.callback(save_and_log_callback, policy_state, critic_state, optimization_metrics)


                return (policy_state, critic_state, state, key), ()
            

            init_carry = (self.policy_state, self.critic_state, state, self.key)
            _, _ = jax.lax.scan(train, init_carry, jnp.arange(self.nr_rollouts))


        state, _ = self.env.reset()
        self.np_state_shape = state.shape
        self.saving_return_buffer = deque(maxlen=100 * self.nr_envs)
        self.evaluation_metrics = {}
        self.last_time = time.time()
        self.global_step = 0
        self.current_nr_updates = 0
        self.current_nr_episodes = 0
        train_loop()


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

    
    def save(self, policy_state, critic_state, type):
        checkpoint = {
            "config_algorithm": self.config.algorithm.to_dict(),
            "policy": policy_state,
            "critic": critic_state
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        if type == "best":
            self.best_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
            os.rename(f"{self.save_path}/tmp/{self.best_model_file_name}", f"{self.save_path}/{self.best_model_file_name}")
            os.remove(f"{self.save_path}/tmp/_METADATA")
            os.rmdir(f"{self.save_path}/tmp")

            if self.track_wandb:
                wandb.save(f"{self.save_path}/{self.best_model_file_name}", base_path=self.save_path)
        elif type == "latest":
            self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
            os.rename(f"{self.save_path}/tmp/{self.latest_model_file_name}", f"{self.save_path}/{self.latest_model_file_name}")
            os.remove(f"{self.save_path}/tmp/_METADATA")
            os.rmdir(f"{self.save_path}/tmp")

            if self.track_wandb:
                wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = "/".join(splitted_path[:-1])
        checkpoint_file_name = splitted_path[-1]

        check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=checkpoint_file_name)
        checkpointer = orbax.checkpoint.Checkpointer(check_point_handler)
        
        # Mask out keys "save_data" and "data_points" in config.algorithm.to_dict()
        algorithm_dict = config.algorithm.to_dict()
        if "save_data" in algorithm_dict:
            del algorithm_dict["save_data"]
        if "data_points" in algorithm_dict:
            del algorithm_dict["data_points"]
        if "use_student" in algorithm_dict:
            del algorithm_dict["use_student"]
        if "dagger_style" in algorithm_dict:
            del algorithm_dict["dagger_style"]
        if "dagger_online" in algorithm_dict:
            del algorithm_dict["dagger_online"]
        for key in (
            "student_model_path",
            "student_checkpoint_dir",
            "student_dataset_path",
            "student_hidden_dims",
            "student_backend",
            "student_batch_size",
            "bootstrap_timesteps",
            "bootstrap_readout",
            "bootstrap_num_sample_iter",
            "bootstrap_sample_period",
            "bootstrap_crossover_epochs",
            "bootstrap_neuron_threshold",
            "bootstrap_current_decay",
            "bootstrap_voltage_decay",
            "bootstrap_weight_scale",
            "bootstrap_weight_norm",
            "student_learning_rate",
            "student_train_epochs",
            "student_num_workers",
            "rollout_policy_stage",
            "snn_enabled",
            "snn_threshold",
            "snn_timesteps",
            "snn_convert_every_iteration",
            "snn_export_dir",
            "record",
            "record_robot_index",
            "record_seconds",
            "record_dir",
            "record_fps",
        ):
            if key in algorithm_dict:
                del algorithm_dict[key]

        # Restore only the configuration to check for missing keys
        restored_struct = checkpointer.restore(checkpoint_dir, item={"config_algorithm": algorithm_dict})
        loaded_algorithm_config = restored_struct['config_algorithm']

        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        
        # # Add missing keys for backward compatibility
        # if "save_data" not in config.algorithm:
        #     config.algorithm.save_data = False
        # if "data_points" not in config.algorithm:
        #     config.algorithm.data_points = 0

        model = PPO(config, env, run_path, writer)

        target = {
            "config_algorithm": algorithm_dict,
            "policy": model.policy_state,
            "critic": model.critic_state
        }
        checkpoint = checkpointer.restore(checkpoint_dir, item=target)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        
        # import pdb; pdb.set_trace()

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_state, state):
            @partial(jax.jit, static_argnums=(1, 2))
            def get_action_single(state, env_id_0, env_id_1):
                state_b = state[env_id_0:env_id_1]

                dynamic_joint_combined_state_b = state_b[:, :self.dynamic_joint_observation_lengths[env_id_0]].reshape((-1, self.nr_dynamic_joint_observations[env_id_0], self.single_dynamic_joint_observation_length))
                dynamic_joint_description_b = dynamic_joint_combined_state_b[:, :, :self.dynamic_joint_description_size]
                dynamic_joint_state_b = dynamic_joint_combined_state_b[:, :, self.dynamic_joint_description_size:]

                dynamic_foot_combined_state_b = state_b[:, self.dynamic_joint_observation_lengths[env_id_0]:self.dynamic_joint_observation_lengths[env_id_0] + self.dynamic_foot_observation_lengths[env_id_0]].reshape((-1, self.nr_dynamic_foot_observations[env_id_0], self.single_dynamic_foot_observation_length))
                dynamic_foot_description_b = dynamic_foot_combined_state_b[:, :, :self.dynamic_foot_description_size]
                dynamic_foot_state_b = dynamic_foot_combined_state_b[:, :, self.dynamic_foot_description_size:]

                action_mean_b, action_logstd_b = self.policy.apply(policy_state.params, dynamic_joint_description_b, dynamic_joint_state_b, dynamic_foot_description_b, dynamic_foot_state_b, state_b[:, self.policy_general_state_mask[env_id_0]])

                action = jnp.concatenate([action_mean_b, jnp.zeros((self.nr_envs_per_env_type, self.missing_nr_of_actions[env_id_0]))], axis=1)
                return self.get_processed_action(action)

            processed_actions = []
            for i in range(self.env_ids.shape[0] - 1):
                env_id_0 = self.env_ids[i]
                env_id_1 = self.env_ids[i + 1]
                processed_actions.append(get_action_single(state, env_id_0, env_id_1))
            return jnp.concatenate(processed_actions, axis=0)

        @partial(jax.jit, static_argnums=(2))
        def get_action_multi_render(policy_state, state, env_id):
            dynamic_joint_combined_state_b = state[:, :self.dynamic_joint_observation_lengths[env_id]].reshape((-1, self.nr_dynamic_joint_observations[env_id], self.single_dynamic_joint_observation_length))
            dynamic_joint_description_b = dynamic_joint_combined_state_b[:, :, :self.dynamic_joint_description_size]
            dynamic_joint_state_b = dynamic_joint_combined_state_b[:, :, self.dynamic_joint_description_size:]

            dynamic_foot_combined_state_b = state[:, self.dynamic_joint_observation_lengths[env_id]:self.dynamic_joint_observation_lengths[env_id] + self.dynamic_foot_observation_lengths[env_id]].reshape((-1, self.nr_dynamic_foot_observations[env_id], self.single_dynamic_foot_observation_length))
            dynamic_foot_description_b = dynamic_foot_combined_state_b[:, :, :self.dynamic_foot_description_size]
            dynamic_foot_state_b = dynamic_foot_combined_state_b[:, :, self.dynamic_foot_description_size:]

            action_mean_b, action_logstd_b = self.policy.apply(policy_state.params, dynamic_joint_description_b, dynamic_joint_state_b, dynamic_foot_description_b, dynamic_foot_state_b, state[:, self.policy_general_state_mask[env_id]])
            action = jnp.concatenate([action_mean_b, jnp.zeros((self.nr_envs_per_env_type, self.missing_nr_of_actions[env_id]))], axis=1)
            return self.get_processed_action(action)

        self.set_eval_mode()

        algorithm_config = self.config.algorithm
        student_checkpoint_dir = getattr(algorithm_config, "student_checkpoint_dir", "/app/one_policy_to_run_them_all/student")
        student_model_path = getattr(algorithm_config, "student_model_path", os.path.join(student_checkpoint_dir, "student_model_best.pth"))
        student_dataset_path = getattr(algorithm_config, "student_dataset_path", os.path.join(student_checkpoint_dir, "teacher_student_dagger_dataset.npz"))
        student_hidden_dims = list(getattr(algorithm_config, "student_hidden_dims", DEFAULT_HIDDEN_DIMS))
        student_backend = getattr(algorithm_config, "student_backend", "ann")
        rollout_policy_stage = getattr(algorithm_config, "rollout_policy_stage", "student")
        snn_enabled = getattr(algorithm_config, "snn_enabled", False)
        snn_threshold = getattr(algorithm_config, "snn_threshold", 0.1)
        snn_timesteps = getattr(algorithm_config, "snn_timesteps", 10)
        snn_convert_every_iteration = getattr(algorithm_config, "snn_convert_every_iteration", True)
        snn_export_dir = getattr(algorithm_config, "snn_export_dir", os.path.join(student_checkpoint_dir, "snn_exports"))
        record_enabled = getattr(algorithm_config, "record", self.record)
        self.record_robot_index = int(getattr(algorithm_config, "record_robot_index", self.record_robot_index))
        self.record_seconds = float(getattr(algorithm_config, "record_seconds", self.record_seconds))
        self.record_fps = int(getattr(algorithm_config, "record_fps", self.record_fps))
        self.record_dir = getattr(algorithm_config, "record_dir", self.record_dir)

        pipeline = PolicyDistillationPipeline(
            input_dim=self.os_shape[0],
            output_dim=self.as_shape[0],
            student_hidden_dims=student_hidden_dims,
            training_config=StudentTrainingConfig(
                dataset_path=student_dataset_path,
                backend=student_backend,
                batch_size=getattr(algorithm_config, "student_batch_size", 64),
                learning_rate=getattr(algorithm_config, "student_learning_rate", 1e-4),
                epochs=getattr(algorithm_config, "student_train_epochs", 60),
                hidden_dims=student_hidden_dims,
                num_workers=getattr(algorithm_config, "student_num_workers", 0),
                checkpoint_dir=student_checkpoint_dir,
                best_checkpoint_name=os.path.basename(student_model_path),
            ),
            snn_config=SNNConversionConfig(
                threshold=snn_threshold,
                timesteps=snn_timesteps,
                export_dir=snn_export_dir,
            ),
            bootstrap_config=BootstrapTrainingConfig(
                dataset_path=student_dataset_path,
                batch_size=getattr(algorithm_config, "student_batch_size", 64),
                learning_rate=getattr(algorithm_config, "student_learning_rate", 1e-4),
                epochs=getattr(algorithm_config, "student_train_epochs", 60),
                hidden_dims=student_hidden_dims,
                num_workers=getattr(algorithm_config, "student_num_workers", 0),
                checkpoint_dir=student_checkpoint_dir,
                best_checkpoint_name=os.path.basename(student_model_path),
                latest_checkpoint_name=os.path.basename(student_model_path).replace("best", "latest"),
                timesteps=getattr(algorithm_config, "bootstrap_timesteps", 3),
                readout=getattr(algorithm_config, "bootstrap_readout", "mean"),
                num_sample_iter=getattr(algorithm_config, "bootstrap_num_sample_iter", 10),
                sample_period=getattr(algorithm_config, "bootstrap_sample_period", 10),
                crossover_epochs=tuple(getattr(algorithm_config, "bootstrap_crossover_epochs", ())),
                neuron_threshold=getattr(algorithm_config, "bootstrap_neuron_threshold", 1.0),
                current_decay=getattr(algorithm_config, "bootstrap_current_decay", 0.25),
                voltage_decay=getattr(algorithm_config, "bootstrap_voltage_decay", 0.03),
                weight_scale=getattr(algorithm_config, "bootstrap_weight_scale", 1.0),
                weight_norm=getattr(algorithm_config, "bootstrap_weight_norm", False),
            ),
        )

        def teacher_predict(policy_state, state, env_id=None):
            if self.multi_render and env_id is not None:
                return get_action_multi_render(policy_state, state, env_id)
            return get_action(policy_state, state)

        controller = PolicyStageController(teacher_predict=teacher_predict, initial_stage=rollout_policy_stage)

        def current_env_id():
            return self.env.active_env_id if self.multi_render else None

        def get_behavior_action(state, env_id_override=None):
            env_id = current_env_id() if env_id_override is None else env_id_override
            return controller.predict(self.policy_state, state, env_id=env_id, role="behavior")

        def get_teacher_label_action(state):
            return controller.predict(self.policy_state, state, env_id=current_env_id(), role="teacher_label")

        def load_student_checkpoint(log_missing=False):
            if not os.path.exists(student_model_path):
                if log_missing:
                    rlx_logger.warning(f"Student model not found at {student_model_path}")
                return False
            pipeline.load_student_policy(student_model_path)
            controller.register_student(pipeline.predict_student)
            rlx_logger.info(f"Loaded student model from {student_model_path}")
            return True

        def build_snn_artifact(export_name=None):
            if student_backend != "bootstrap" and not snn_enabled:
                return False
            try:
                pipeline.build_snn_policy(student_model=pipeline.student_model)
                controller.register_snn(pipeline.predict_snn)
            except Exception as exc:
                rlx_logger.warning(f"Could not build SNN policy: {exc}")
                return False

            if student_backend == "bootstrap":
                if export_name and snn_export_dir:
                    rlx_logger.info("Bootstrap backend does not export HDF5 artifacts; skipping export.")
                return True

            if export_name and snn_export_dir:
                try:
                    export_path = os.path.join(snn_export_dir, export_name)
                    pipeline.snn_policy.export(export_path)
                    rlx_logger.info(f"Exported SNN model to {export_path}")
                except Exception as exc:
                    rlx_logger.warning(f"Built SNN policy but failed to export it: {exc}")
            return True

        if self.use_student or self.dagger_style or self.dagger_online or snn_enabled or rollout_policy_stage in {"student", "snn"}:
            if load_student_checkpoint(log_missing=self.use_student or self.dagger_style or self.dagger_online):
                controller.set_stage("student")
                if rollout_policy_stage == "snn" and build_snn_artifact():
                    controller.set_stage("snn")
            else:
                controller.set_stage("teacher")

        if record_enabled:
            output_paths = self._record_policy_videos(get_behavior_action, controller.stage)
            for output_path in output_paths:
                rlx_logger.info(f"Saved recording to {output_path}")
            print(f"Recording finished. Saved {len(output_paths)} video(s) to {self.record_dir} with simulation-synced timing.")
            return

        if self.dagger_online:
            rlx_logger.info(f"Starting DAgger online training for {self.dagger_online} iterations")
            if pipeline.student_model is None:
                pipeline.build_student_policy()
                controller.register_student(pipeline.predict_student)

            states_buffer = []
            actions_buffer = []
            for iteration in range(self.dagger_online):
                rlx_logger.info(f"DAgger Iteration {iteration + 1}/{self.dagger_online}")
                state, _ = self.env.reset()
                episode_return = 0
                states = []
                actions = []
                total_iterations = self.data_points // self.nr_envs

                if iteration == 0:
                    expert_ratio = 1.0
                elif iteration == 1:
                    expert_ratio = 0.5
                elif iteration == 2:
                    expert_ratio = 0.25
                else:
                    expert_ratio = 0.0

                expert_data_points = int(expert_ratio * total_iterations)
                controller.set_stage("teacher")
                for _ in range(expert_data_points):
                    processed_action = get_teacher_label_action(state)
                    states.append(np.array(state, copy=True))
                    actions.append(np.array(jax.device_get(processed_action), copy=True))
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                    episode_return += reward

                controller.set_stage(rollout_policy_stage if rollout_policy_stage in {"student", "snn"} else "student")
                student_data_points = total_iterations - expert_data_points
                for _ in range(student_data_points):
                    processed_action_expert = get_teacher_label_action(state)
                    states.append(np.array(state, copy=True))
                    actions.append(np.array(jax.device_get(processed_action_expert), copy=True))
                    processed_action_behavior = get_behavior_action(state)
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action_behavior))
                    episode_return += reward

                states = np.concatenate(states, axis=0)
                actions = np.concatenate(actions, axis=0)
                states_buffer.append(states)
                actions_buffer.append(actions)
                rlx_logger.info(f"Iteration {iteration + 1} - Return: {episode_return}")

                states_train = np.concatenate(states_buffer, axis=0)
                actions_train = np.concatenate(actions_buffer, axis=0)
                rlx_logger.info(f"Collected {states_train.shape[0]} samples for iteration {iteration + 1}")

                artifacts = pipeline.train_on_aggregated_data(
                    states=states_train,
                    actions=actions_train,
                    iteration=iteration + 1,
                )
                controller.register_student(pipeline.predict_student)
                controller.set_stage("student")
                rlx_logger.info(
                    f"Student update {iteration + 1}: train_loss={artifacts.last_train_loss:.6f}, "
                    f"val_loss={artifacts.last_val_loss:.6f}, best_val={artifacts.best_val_loss:.6f}"
                )

                if snn_enabled and (snn_convert_every_iteration or controller.snn_predict is None or rollout_policy_stage == "snn"):
                    export_name = f"student_model_dagger_iter_{iteration + 1}.hdf5"
                    if build_snn_artifact(export_name=export_name) and rollout_policy_stage == "snn":
                        controller.set_stage("snn")

            final_data_dir = os.path.dirname(student_dataset_path)
            if final_data_dir:
                os.makedirs(final_data_dir, exist_ok=True)
            states_final = np.concatenate(states_buffer, axis=0)
            actions_final = np.concatenate(actions_buffer, axis=0)
            np.savez(student_dataset_path, states=states_final, actions=actions_final)
            rlx_logger.info(f"Saved aggregated DAgger dataset ({states_final.shape[0]} samples) to {student_dataset_path}")
            return

        for i in range(episodes):
            episode_return = 0
            state, _ = self.env.reset()
            if self.save_data:
                states = []
                actions = []
                total_iterations = self.data_points // self.nr_envs
                if self.dagger_style:
                    expert_data_points = int(0.3 * total_iterations)
                    controller.set_stage("teacher")
                    for _ in range(expert_data_points):
                        processed_action = get_teacher_label_action(state)
                        states.append(np.array(state, copy=True))
                        actions.append(np.array(jax.device_get(processed_action), copy=True))
                        state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                        episode_return += reward

                    controller.set_stage(rollout_policy_stage if rollout_policy_stage in {"student", "snn"} else "student")
                    dagger_data_points = total_iterations - expert_data_points
                    for _ in range(dagger_data_points):
                        processed_action_behavior = get_behavior_action(state)
                        state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action_behavior))
                        episode_return += reward
                        processed_action_expert = get_teacher_label_action(state)
                        states.append(np.array(state, copy=True))
                        actions.append(np.array(jax.device_get(processed_action_expert), copy=True))
                else:
                    for _ in range(total_iterations):
                        processed_action = get_behavior_action(state)
                        states.append(np.array(state, copy=True))
                        actions.append(np.array(jax.device_get(processed_action), copy=True))
                        state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                        episode_return += reward

                states = np.concatenate(states, axis=0)
                actions = np.concatenate(actions, axis=0)
                output_dir = "teacher-student"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "teacher_dataset.npz")
                np.savez(output_path, states=states, actions=actions)
                print(f"Saved {states.shape[0]} samples to {output_path}")
            else:
                while True:
                    processed_action = get_behavior_action(state)
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                    episode_return += reward
                    done = np.logical_or(terminated, truncated)
                    if np.all(done):
                        break
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
