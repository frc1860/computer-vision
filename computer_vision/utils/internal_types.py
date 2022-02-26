from dataclasses import dataclass

import numpy as np


@dataclass
class Resolution:
    width: int = 0
    height: int = 0


@dataclass
class Position:
    x: float = 0
    y: float = 0


@dataclass
class Range:
    min: float = 0
    max: float = 255


@dataclass
class HsvRange:
    hue: Range = Range()
    saturation: Range = Range()
    value: Range = Range()


@dataclass
class TargetContourFilterParameters:
    area: Range = Range(min=700, max=100000)
    perimeter: Range = Range(min=0, max=100000)
    width: Range = Range(min=80, max=750)
    height: Range = Range(min=50, max=400)
    solidity: Range = Range(min=0, max=100)
    vertex_count: Range = Range(min=0, max=10500)
    ratio: Range = Range(min=0, max=1000)


@dataclass
class TargetDistanceParameters:
    a: float = 0
    b: float = 0
    c: float = 0


@dataclass
class BallDistanceParameters:
    focal_length: float = 0
    ball_diameter: float = 0


@dataclass
class TargetImageProcessingResponse:
    found: bool
    image_with_target: np.ndarray
    binary_image: np.ndarray
    target_position: Position


@dataclass
class BallImageProcessingResponse:
    found: bool
    image_with_ball: np.ndarray
    binary_image: np.ndarray
    ball_position: Position
    ball_diameter: float
