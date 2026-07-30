import numpy as np


class PlaneTerrainGeneration:
    def __init__(self, env):
        self.env = env
        self.xml_file_name = "vr_m3_1_12dof.xml"
        self.center_height = 0.0
        self.nr_sampled_heights = 1
        self.current_difficulty_level = 0.0
        self.sampled_heights = np.zeros(self.nr_sampled_heights)
    
    def step(self, obs, reward, absorbing, info):
        return
    
    def sample(self):
        return
