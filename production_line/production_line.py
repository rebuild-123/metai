import math
import random
import typing

from .production_line_units import Box, Lidar

class Production_Line:
    def __init__(self, length: float, velocity: float, box_side_length: float):
        self.length = length
        self.velocity = velocity
        self.box_side_length = box_side_length

        self.lidars = []
        self.boxes = []
        self.occupied = [False]*math.ceil(self.length)

    def check_if_it_is_occupied(self, place: float) -> bool:
        place = math.floor(place) - (place%1 == 0)
        return self.occupied[place]

    def add_lidar(self, lidar: Lidar) -> None:
        self.lidars.append(lidar)

    def add_box(self, box: Box) -> None:
        self.boxes = [box] + self.boxes
        left, right = box.get_edges()
        left, right = math.floor(left), math.floor(right) + (right%1 != 0)
        for idx in range(left, right): self.occupied[idx] = True

    def remove_last_box(self) -> Box:
        box = self.boxes.pop(-1)
        left, right = box.get_edges()
        left, right = math.floor(left), math.floor(right) + (right%1 != 0)
        for idx in range(left, right): self.occupied[idx] = False
        box.reset()
        return box

    @property
    def head_lidar_signal(self) -> bool:
        return self.lidars[0].scan(self)

    @property
    def tail_lidar_signal(self) -> bool:
        return self.lidars[-1].scan(self)

    def shift(self) -> None:
        self.occupied = [False]*math.ceil(self.length)
        for b in self.boxes:
            b.update_place(self.velocity)
            left, right = b.get_edges()
            left, right = math.floor(left), math.floor(right) + (right%1 != 0)
            for idx in range(left, right): self.occupied[idx] = True