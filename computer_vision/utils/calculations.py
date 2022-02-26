from math import tan

from computer_vision.utils.internal_types import (
    BallDistanceParameters,
    TargetDistanceParameters,
)


def calculate_horizontal_angle(
    x: float, image_width: float, camera_offset: float
) -> float:
    angle = (x - image_width / 2) * 60 / image_width
    return angle + camera_offset


def calculate_target_distance(y: float, params: TargetDistanceParameters) -> float:
    return params.a / tan(params.b * y + params.c)


def calculate_ball_distance(
    ball_diameter: float, params: BallDistanceParameters
) -> float:
    return params.ball_diameter * params.focal_length / ball_diameter


def calculate_launcher_angle(distance: float) -> float:
    # TODO: Implement launcher angle calculation
    return 0
