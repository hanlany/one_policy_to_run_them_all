import numpy as np


GENERAL_STATE_FOR_POLICY_NAMES = (
    "trunk_roll_velocity",
    "trunk_pitch_velocity",
    "trunk_yaw_velocity",
    "goal_x_velocity",
    "goal_y_velocity",
    "goal_yaw_velocity",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    "p_gain",
    "d_gain",
    "action_scaling_factor",
    "mass",
    "robot_length",
    "robot_width",
    "robot_height",
)

GENERAL_STATE_FOR_CRITIC_NAMES = (
    "trunk_x_velocity",
    "trunk_y_velocity",
    "trunk_z_velocity",
    "trunk_roll_velocity",
    "trunk_pitch_velocity",
    "trunk_yaw_velocity",
    "goal_x_velocity",
    "goal_y_velocity",
    "goal_yaw_velocity",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    "height_0",
    "p_gain",
    "d_gain",
    "action_scaling_factor",
    "mass",
    "robot_length",
    "robot_width",
    "robot_height",
)


def build_general_state_mask(observation_name_to_ids, observation_shape, names):
    mask = np.zeros(observation_shape, dtype=bool)
    for env_id, name_to_id in enumerate(observation_name_to_ids):
        for name in names:
            mask[env_id, name_to_id[name]] = True
    return mask
