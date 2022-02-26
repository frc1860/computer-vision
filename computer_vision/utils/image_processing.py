import typing

import cv2
import numpy as np

from .internal_types import (
    BallImageProcessingResponse,
    HsvRange,
    Position,
    Range,
    Resolution,
    TargetContourFilterParameters,
    TargetImageProcessingResponse,
)


class CouldNotGetFrame(Exception):
    pass


def get_available_cameras() -> typing.List[cv2.VideoCapture]:
    # TODO: Improve camera identification algorithm
    available_indexes = [0, 2]
    captures = [cv2.VideoCapture(index) for index in available_indexes]
    return captures


def get_frame(capture: cv2.VideoCapture) -> np.ndarray:
    success, frame = capture.read()
    if not success:
        raise CouldNotGetFrame()
    return frame


def resize(frame: np.ndarray, resolution: Resolution) -> np.ndarray:
    return cv2.resize(frame, (resolution.width, resolution.height))


def set_brightness(capture: cv2.VideoCapture, brightness: float) -> None:
    capture.set(cv2.CAP_PROP_BRIGHTNESS, brightness)


def set_exposure(capture: cv2.VideoCapture, exposure: float) -> None:
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    # capture.set(cv2.CAP_PROP_EXPOSURE, exposure)


def set_fps(capture: cv2.VideoCapture, fps: int) -> None:
    capture.set(cv2.CAP_PROP_FPS, fps)


def filter_contour_by_area(
    min_area: float,
) -> typing.Callable[[typing.List[typing.Any]], bool]:
    def func(contour: typing.List[typing.Any]) -> bool:
        area = cv2.contourArea(contour)
        return area >= min_area

    return func


def filter_contour_by_perimeter(
    min_perimeter: float,
) -> typing.Callable[[typing.List[typing.Any]], bool]:
    def func(contour: typing.List[typing.Any]) -> bool:
        perimeter = cv2.arcLength(contour, True)
        return perimeter >= min_perimeter

    return func


def filter_contour_by_size(
    width: Range,
    height: Range,
    ratio: Range,
) -> typing.Callable[[typing.List[typing.Any]], bool]:
    def func(contour: typing.List[typing.Any]) -> bool:
        _, __, w, h = cv2.boundingRect(contour)
        r = w / h
        return (
            w >= width.min
            and w <= width.max
            and h >= height.min
            and h <= height.max
            and r >= ratio.min
            and r <= ratio.max
        )

    return func


def filter_contour_by_solidity(
    solidity: Range,
) -> typing.Callable[[typing.List[typing.Any]], bool]:
    def func(contour: typing.List[typing.Any]) -> bool:
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        solid = 100000000
        try:
            solid = 100 * area / cv2.contourArea(hull)
        except Exception:
            pass
        return solid >= solidity.min and solid <= solidity.max

    return func


def filter_contour_by_vertex_count(
    vertex_count: Range,
) -> typing.Callable[[typing.List[typing.Any]], bool]:
    def func(contour: typing.List[typing.Any]) -> bool:
        return len(contour) >= vertex_count.min and len(contour) <= vertex_count.max

    return func


def calculate_contour_offset(
    frame_width: float,
) -> typing.Callable[
    [typing.List[typing.Any]], typing.Tuple[float, typing.List[typing.Any]]
]:
    def func(
        contour: typing.List[typing.Any],
    ) -> typing.Tuple[float, typing.List[typing.Any]]:
        pos = calculate_contour_center(contour)
        return (abs(frame_width / 2 - pos.x), contour)

    return func


def calculate_contour_center(contour: typing.Any) -> Position:
    x, y, w, h = cv2.boundingRect(contour)
    return Position(x=int(x + w / 2), y=int(y + h / 2))


def process_target_image(
    frame: np.ndarray,
    hsv_range: HsvRange,
    contour_filter_parameters: TargetContourFilterParameters,
) -> TargetImageProcessingResponse:

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

    # Filtering color
    mask = cv2.inRange(hsv_image, lower_hsv_threshold, upper_hsv_threshold)

    _, contours, __ = cv2.findContours(
        mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours_by_area = list(
        filter(filter_contour_by_area(contour_filter_parameters.area.min), contours)
    )
    filtered_contours_by_perimeter = list(
        filter(
            filter_contour_by_perimeter(contour_filter_parameters.perimeter.min),
            filtered_contours_by_area,
        )
    )
    filtered_contours_by_size = list(
        filter(
            filter_contour_by_size(
                contour_filter_parameters.width,
                contour_filter_parameters.height,
                contour_filter_parameters.ratio,
            ),
            filtered_contours_by_perimeter,
        )
    )
    filtered_contours_by_solidity = list(
        filter(
            filter_contour_by_solidity(contour_filter_parameters.solidity),
            filtered_contours_by_size,
        )
    )
    filtered_contours = list(
        filter(
            filter_contour_by_vertex_count(contour_filter_parameters.vertex_count),
            filtered_contours_by_solidity,
        )
    )

    biggest_contours = list(
        map(
            lambda elem: elem[1],
            sorted(
                map(calculate_contour_offset(frame.shape[1]), filtered_contours),
                key=lambda elem: elem[0],
            ),
        )
    )

    if len(biggest_contours) > 0:
        biggest_contour = biggest_contours[0]
        center = calculate_contour_center(biggest_contour)

        # Drawing on the image where the code detected the target
        output_image = frame.copy()
        cv2.circle(output_image, (center.x, center.y), 7, (0, 0, 255), 5)
        cv2.drawContours(output_image, biggest_contours, 0, (255, 255, 0), 2)

        return TargetImageProcessingResponse(
            found=True,
            image_with_target=output_image,
            binary_image=mask,
            target_position=center,
        )

    return TargetImageProcessingResponse(
        found=False,
        image_with_target=frame,
        binary_image=mask,
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
        ball_diameter=radius * 2,
    )
