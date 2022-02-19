from math import atan, pi, tan

from computer_vision.utils.internal_types import (
    BallDistanceParameters,
    TargetDistanceParameters,
)


def calculate_horizontal_angle(
    x: float, image_width: float, focal_length: float, camera_offset: float
) -> float:
    angle = atan((x - (image_width - 1) / 2) / focal_length) * 180 / pi
    return angle + camera_offset


def calculate_target_distance(y: float, params: TargetDistanceParameters) -> float:
    return params.a / tan(params.b * y + params.c)


def calculate_ball_distance(
    ball_diameter: float, params: BallDistanceParameters
) -> float:
    # TODO: Implement ball distance calculation
    return 0


def calculate_launcher_angle(distance: float) -> float:
    # TODO: Implement launcher angle calculation
    return 0
