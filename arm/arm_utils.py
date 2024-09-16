import json
import math
import random
import typing


def calculate_spatial_position(
    prefix: str, arm_1_length: float, arm_2_length: float, angle_1_degree: float, angle_2_degree: float
) -> dict[str, float]:
    rad = math.pi/180
    x = abs(arm_2_length*math.sin(rad*angle_2_degree))*math.cos(rad*angle_1_degree)
    y = abs(arm_2_length*math.sin(rad*angle_2_degree))*math.sin(rad*angle_1_degree)
    z = arm_1_length - arm_2_length*math.cos(rad*angle_2_degree)
    return {prefix + '_x': x, prefix + '_y': y, prefix + '_z': z}

def generate_random_action( max_variable_arm_length_unit: int, max_variable_angle_unit: int) -> list[int]:
    return [
        random.randrange(0, max_variable_arm_length_unit+1), # modified_unit_for_arm_1
        random.randrange(0, max_variable_arm_length_unit+1), # modified_unit_for_arm_2
        random.randrange(0, max_variable_angle_unit+1), # modified_unit_for_angle_1
        random.randrange(0, max_variable_angle_unit+1), # modified_unit_for_angle_2
    ]

def generate_random_parameters(
    arm_1_range: list[float, float], arm_2_range: list[float, float], 
    min_arm_zoom_length: float, min_angle_degree: float,
) -> list[float]:
    return [
        arm_1_range[0] + min_arm_zoom_length*(random.uniform(0, arm_1_range[1] - arm_1_range[0])//min_arm_zoom_length),  # arm_1_length
        arm_2_range[0] + min_arm_zoom_length*(random.uniform(0, arm_2_range[1] - arm_2_range[0])//min_arm_zoom_length),  # arm_2_length
        min_angle_degree*(random.uniform(0,360)//min_angle_degree), # angle_1_degree
        min_angle_degree*(random.uniform(0,360)//min_angle_degree), # angle_2_degree
    ]

def generate_random_target(
    arm_1_range: list[float, float], arm_2_range: list[float, float], 
    min_arm_zoom_length: float, min_angle_degree: float,
) -> dict[str, float]:
    paras = generate_random_parameters(
        arm_1_range=arm_1_range, arm_2_range=arm_2_range, 
        min_arm_zoom_length=min_arm_zoom_length, min_angle_degree=min_angle_degree,
    )
    return calculate_spatial_position(
        prefix='target',
        arm_1_length=paras[0], arm_2_length=paras[1], 
        angle_1_degree=paras[2], angle_2_degree=paras[3],
    )

def calculate_distance(position: dict[str, float], target: dict[str, float]) -> float:
    x = (position['position_x'] - target['target_x'])**2
    y = (position['position_y'] - target['target_y'])**2
    z = (position['position_z'] - target['target_z'])**2
    return (x + y + z)**0.5

def calculate_reward(
    pre_position: dict[str, float], position: dict[str, float], target: dict[str, float],
    reward_discount_factor: float, 
    arm_1_length: float, arm_2_length: float,
    max_arm_1_length: float, max_arm_2_length: float,
) -> float:
    pre_distance = calculate_distance(position=pre_position, target=target)
    cur_distance = calculate_distance(position=position, target=target)
    reward = (pre_distance - reward_discount_factor*cur_distance)/(2*max_arm_2_length)
    # reward = reward - (10 if max_arm_1_length < arm_1_length else 0) - (10 if max_arm_2_length < arm_2_length else 0)
    # print(pre_distance, cur_distance, reward)
    return reward

def confirm_whether_it_is_over(
    position: dict[str, float], target: dict[str, float], max_arm_2_length: float, termination_condition: float,
) -> bool:
    distance = calculate_distance(position=position, target=target)
    percentage = distance/(2*max_arm_2_length)
    #print(percentage)
    return True if percentage <= termination_condition else False