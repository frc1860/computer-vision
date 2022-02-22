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


def set_brightness(capture: cv2.VideoCapture, brightness: float) -> None:
    capture.set(cv2.CAP_PROP_BRIGHTNESS, brightness)


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

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hsv_threshold = np.array(
        [
            hsv_range.hue.min,
            hsv_range.saturation.min,
            hsv_range.value.min,
        ]
    )
    upper_hsv_threshold = np.array(
        [
            hsv_range.hue.max,
            hsv_range.saturation.max,
            hsv_range.value.max,
        ]
    )

    # TODO: Stop hardcoding these erosion and dilation parameters

    # Removing noise
    mask = cv2.inRange(hsv_image, lower_hsv_threshold, upper_hsv_threshold)
    erosion_kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, erosion_kernel, iterations=1)

    # Increasing found area
    dilation_kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, dilation_kernel, iterations=10)

    # Converting to grayscale and applying blur
    gray_filtered_image = cv2.GaussianBlur(
        cv2.cvtColor(
            cv2.bitwise_and(frame, frame, mask=dilated_mask), cv2.COLOR_BGR2GRAY
        ),
        (5, 5),
        0,
    )

    # Finding circles
    detected_circles = cv2.HoughCircles(
        gray_filtered_image,
        cv2.HOUGH_GRADIENT,
        1,
        75,
        param1=45,
        param2=45,
        minRadius=10,
        maxRadius=400,
    )

    if detected_circles is None:
        return BallImageProcessingResponse(
            found=False,
            image_with_ball=frame,
            binary_image=dilated_mask,
            ball_position=Position(x=-1, y=-1),
            ball_diameter=0,
        )

    detected_circles_fixed = np.uint16(np.around(detected_circles))

    # Will consider the biggest ball the correct one
    radius = 0
    for circle in detected_circles_fixed[0, :]:
        x, y, current_radius = circle[0], circle[1], circle[2]
        if current_radius > radius:
            radius = current_radius
            ball_position = Position(x=x, y=y)

    # Drawing on the image where the code detected a ball
    frame_to_draw = frame
    cv2.circle(
        frame_to_draw, (ball_position.x, ball_position.y), radius, (0, 255, 0), 2
    )
    cv2.circle(frame_to_draw, (ball_position.x, ball_position.y), 1, (0, 0, 255), 3)

    return BallImageProcessingResponse(
        found=True,
        image_with_ball=frame_to_draw,
        binary_image=dilated_mask,
        ball_position=ball_position,
        ball_diameter=0,
    )
