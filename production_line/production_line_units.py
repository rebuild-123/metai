import math
import random
import typing


class Lidar:
    def __init__(self, place: float):
        self.place = place

    def scan(self, production_line) -> bool:
        return production_line.check_if_it_is_occupied(self.place)

class Box:
    def __init__(self, box_side_length: float):
        self.box_side_length = box_side_length
        self.left_edge_place = 0.0
        self.right_edge_place = self.box_side_length

    def update_place(self, step: float) -> None:
        self.left_edge_place += step
        self.right_edge_place += step

    def get_edges(self) -> tuple[float, float]:
        return self.left_edge_place, self.right_edge_place

    def reset(self) -> None:
        self.left_edge_place = 0.0
        self.right_edge_place = self.box_side_length