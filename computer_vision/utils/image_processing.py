import os
import typing

import cv2
import numpy as np

from .internal_types import (
    BallImageProcessingResponse,
    HsvRange,
    Position,
    Resolution,
    TargetImageProcessingResponse,
)


class CouldNotGetFrame(Exception):
    pass


def get_available_cameras() -> typing.List[cv2.VideoCapture]:
    available_indexes = [
        int(file[-1]) for file in os.listdir("/dev") if "video" in file
    ]
    # TODO: Improve camera identification algorithm
    available_indexes = [0, 2]
    captures = [cv2.VideoCapture(index) for index in available_indexes]
    available_captures = [capture for capture in captures if capture.read()[0]]
    return available_captures


def get_frame(capture: cv2.VideoCapture) -> np.ndarray:
    success, frame = capture.read()
    if not success:
        raise CouldNotGetFrame()
    return frame


def resize(frame: np.ndarray, resolution: Resolution) -> np.ndarray:
    return cv2.resize(frame, (resolution.width, resolution.height))


def process_target_image(
    frame: np.ndarray, hsv_range: HsvRange
) -> TargetImageProcessingResponse:
    # TODO: Re-implement target identification
    return TargetImageProcessingResponse(
        found=False,
        image_with_target=frame,
        binary_image=frame,
        target_position=Position(x=0, y=0),
    )


def process_ball_image(
    frame: np.ndarray, hsv_range: HsvRange
) -> BallImageProcessingResponse:
    # TODO: Implement ball identification
    return BallImageProcessingResponse(
        found=False,
        image_with_ball=frame,
        binary_image=frame,
        ball_position=Position(x=0, y=0),
        ball_diameter=0,
    )
