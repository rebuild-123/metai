import json
import math
import random
import typing

from .arm_utils import (
    calculate_spatial_position, generate_random_action, calculate_distance, calculate_reward, confirm_whether_it_is_over,
)

def calculate_spatial_position(arm_1_length: float, arm_2_length: float, angle_1_degree: float, angle_2_degree: float) -> dict[str, float]:
    rad = math.pi/180
    x = abs(arm_2_length*math.sin(rad*angle_2_degree))*math.cos(rad*angle_1_degree)
    y = abs(arm_2_length*math.sin(rad*angle_2_degree))*math.sin(rad*angle_1_degree)
    z = arm_1_length - arm_2_length*math.cos(rad*angle_2_degree)
    return {'x': x, 'y': y, 'z': z}

def generate_random_action(
    max_variable_arm_length_unit: int, max_variable_arm_length_unit: int,
    max_variable_angle_unit: int, max_variable_angle_unit: int,
) -> dict[str, int]:
    return {
        'modified_unit_for_arm_1': random.randrange(0, max_variable_arm_length_unit+1),
        'modified_unit_for_arm_2': random.randrange(0, max_variable_arm_length_unit+1),
        'modified_unit_for_angle_1': random.randrange(0, max_variable_angle_unit+1),
        'modified_unit_for_angle_2': random.randrange(0, max_variable_angle_unit+1),
    }

def calculate_distance(position: dict[str, float], target: dict[str, float]) -> float:
    x = (position['x'] - target['x'])**2
    y = (position['y'] - target['y'])**2
    z = (position['z'] - target['z'])**2
    return (x + y + z)**0.5

def calculate_reward(
    pre_position: dict[str, float], position: dict[str, float], target: dict[str, float],
    reward_discount_factor: float, arm_2_length: float,
) -> float:
    pre_distance = calculate_distance(position=pre_position, target=target)
    cur_distance = calculate_distance(position=position, target=target)
    return (pre_distance - reward_discount_factor*cur_distance)/(2*arm_2_length)

def confirm_whether_it_is_over(
    position: dict[str, float], target: dict[str, float], arm_2_length: float, termination_condition: float,
) -> bool:
    distance = calculate_distance(position=position, target=target)
    percentage = distance/(2*arm_2_length)
    return True if percentage <= termination_condition else False