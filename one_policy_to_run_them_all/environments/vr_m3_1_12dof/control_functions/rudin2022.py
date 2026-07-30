class Rudin2022Control:
    def __init__(self, env, control_frequency_hz=50, p_gain=60, d_gain=2.0, scaling_factor=0.75):
        self.env = env
        self.control_frequency_hz = control_frequency_hz
        self.p_gains = self._per_leg([150.0, 150.0, 120.0, 200.0, 200.0, 200.0])
        self.d_gains = self._per_leg([25.0, 25.0, 4.0, 10.0, 8.0, 8.0])
        effort_limits = self._per_leg([360.0, 360.0, 130.0, 360.0, 120.0, 120.0])
        self.scaling_factors = 0.25 * effort_limits / self.p_gains
        self.p_gain = float(self.p_gains.mean())
        self.d_gain = float(self.d_gains.mean())
        self.scaling_factor = float(self.scaling_factors.mean())
        self.add_p_gain = 0.0  # Set by seen robot function
        self.add_d_gain = 0.0  # Set by seen robot function
        self.add_scaling_factor = 0.0  # Set by seen robot function
        self.p_gain_noise_factor = 1.0  # Set by unseen robot function
        self.d_gain_noise_factor = 1.0  # Set by unseen robot function
        self.motor_strength_noise_factor = 1.0  # Set by unseen robot function
        self.joint_position_offset = 0.0 # Set by unseen robot function
        self.seen_p_gain = p_gain + self.add_p_gain
        self.seen_d_gain = d_gain + self.add_d_gain
        self.seen_scaling_factor = scaling_factor + self.add_scaling_factor

    def process_action(self, action):
        self.seen_p_gain = self.p_gain + self.add_p_gain
        self.seen_d_gain = self.d_gain + self.add_d_gain
        self.seen_scaling_factor = self.scaling_factor + self.add_scaling_factor

        scaled_action = action * self.scaling_factors
        target_joint_positions = self.env.nominal_joint_positions + scaled_action
        torques = self.p_gains * self.p_gain_noise_factor * (target_joint_positions - self.env.data.qpos[7:] + self.joint_position_offset) \
                  - self.d_gains * self.d_gain_noise_factor * self.env.data.qvel[6:]
        
        return torques * self.motor_strength_noise_factor

    @staticmethod
    def _per_leg(values):
        import numpy as np

        return np.asarray(values * 2, dtype=float)
