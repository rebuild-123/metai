import json
import math
import random
import typing

import gym


from .arm_utils import (
    calculate_spatial_position, calculate_distance, generate_random_parameters, generate_random_target,
    calculate_reward, confirm_whether_it_is_over,
)

class Arm(gym.Env):
    def __init__(
        self, arm_1_range: list[float, float], arm_2_range: list[float, float], 
        min_arm_zoom_length: float, min_angle_degree: float, 
        max_variable_arm_length_unit: float, max_variable_angle_unit: float,
        reward_discount_factor: float, termination_condition: float,
    ):
        self.arm_1_range = arm_1_range
        self.arm_2_range = arm_2_range
        self.min_arm_zoom_length = min_arm_zoom_length 
        self.min_angle_degree = min_angle_degree
        self.max_variable_angle_unit = max_variable_angle_unit
        self.max_variable_arm_length_unit = max_variable_arm_length_unit
        self.reward_discount_factor = reward_discount_factor
        self.termination_condition = termination_condition

        self.action_space = gym.spaces.MultiDiscrete([
            self.max_variable_arm_length_unit, self.max_variable_arm_length_unit,
            self.max_variable_angle_unit, self.max_variable_angle_unit,
        ])
        self.observation_space = gym.spaces.Dict({
            'arm_1_length': gym.spaces.Box(low=self.arm_1_range[0], high=sum(self.arm_1_range), dtype=float),
            'arm_2_length': gym.spaces.Box(low=self.arm_2_range[0], high=sum(self.arm_2_range), dtype=float),
            'angle_1_degree': gym.spaces.Box(low=0, high=360, dtype=float),
            'angle_2_degree': gym.spaces.Box(low=0, high=360, dtype=float),
            'position_x': gym.spaces.Box(low=0, high=self.arm_2_range[1], dtype=float),
            'position_y': gym.spaces.Box(low=0, high=self.arm_2_range[1], dtype=float),
            'position_z': gym.spaces.Box(low=0, high=self.arm_1_range[1] + self.arm_2_range[1], dtype=float),
            'target_x': gym.spaces.Box(low=0, high=self.arm_2_range[1], dtype=float),
            'target_y': gym.spaces.Box(low=0, high=self.arm_2_range[1], dtype=float),
            'target_z': gym.spaces.Box(low=0, high=self.arm_1_range[1] + self.arm_2_range[1], dtype=float),
        })
        
        self.arm_1_length = self.arm_1_range[0]
        self.arm_2_length = self.arm_2_range[0]
        self.angle_1_degree = 0.0
        self.angle_2_degree = 0.0
        self.position = {'position_x': 0, 'position_y': 0, 'position_z': 0}
        self.target = {'target_x': 0, 'target_y': 0, 'target_z': 0}

    def seed(self, num: float = None) -> None:
        if num: return [num]
        return [random.random()]

    def render(self) -> None:
        pass

    def get_parameters(self) -> dict[str, float]:
        return {
            'arm_1_length': self.arm_1_length, 'arm_2_length': self.arm_2_length,
            'angle_1_degree': self.angle_1_degree, 'angle_2_degree': self.angle_2_degree,
        }

    def generate_observation(self) -> dict[str, typing.Union[float, dict[str, float]]]:
        paras = self.get_parameters()
        temp = paras | self.position | self.target
        return {k: np.array([v], dtype=int)  for k,v in temp.items()}

    def reset(self) -> dict[str, typing.Union[float, dict[str, float]]]:
        paras = generate_random_parameters(
            arm_1_range=self.arm_1_range, arm_2_range=self.arm_2_range, 
            min_arm_zoom_length=self.min_arm_zoom_length, min_angle_degree=self.min_angle_degree,
        )
        self.arm_1_length = paras[0] # arm_1_length
        self.arm_2_length = paras[1] # arm_2_length
        self.angle_1_degree = paras[2] # angle_1_degree
        self.angle_2_degree = paras[3] # angle_2_degree
        self.position = calculate_spatial_position(
            prefix='position',
            arm_1_length=self.arm_1_length, arm_2_length=self.arm_2_length, 
            angle_1_degree=self.angle_1_degree, angle_2_degree=self.angle_2_degree
        )
        self.target = generate_random_target(
            arm_1_range=self.arm_1_range, arm_2_range=self.arm_2_range, 
            min_arm_zoom_length=self.min_arm_zoom_length, min_angle_degree=self.min_angle_degree,
        )
        return self.generate_observation()

    def update_parameters(self, action: dict[str, int]) -> None:
        self.arm_1_length += self.min_arm_zoom_length*action[0]
        self.arm_2_length += self.min_arm_zoom_length*action[1]
        self.angle_1_degree += self.min_angle_degree*action[2]
        self.angle_2_degree += self.min_angle_degree*action[3]

    def step(self, action: dict[str, int]):
        self.update_parameters(action=action)
        pre_position = self.position
        position = self.position = calculate_spatial_position('position', **self.get_parameters())
        obs = self.generate_observation()
        reward = calculate_reward(
            pre_position=pre_position, position=position, target=self.target, 
            reward_discount_factor=self.reward_discount_factor, 
            arm_1_length=self.arm_1_length, arm_2_length=self.arm_2_length,
            max_arm_1_length=self.arm_1_range[1], max_arm_2_length=self.arm_2_range[1],
        )
        done = confirm_whether_it_is_over(
            position=self.position, target=self.target, arm_2_length=self.arm_2_length, 
            termination_condition=self.termination_condition,
        )
        return obs, reward + done, done, {}