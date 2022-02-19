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
