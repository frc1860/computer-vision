import os
import typing

import cv2

from computer_vision.robot_interface.robot_interface import RobotInterface
from computer_vision.utils.calculations import (
    calculate_ball_distance,
    calculate_horizontal_angle,
    calculate_launcher_angle,
    calculate_target_distance,
)
from computer_vision.utils.fake_capture import FakeCapture
from computer_vision.utils.image_processing import (
    get_available_cameras,
    process_ball_image,
    process_target_image,
    resize,
    set_brightness,
    set_exposure,
    set_fps,
)
from computer_vision.utils.internal_types import (
    BallDistanceParameters,
    HsvRange,
    TargetContourFilterParameters,
    TargetDistanceParameters,
)


class InvalidRobotInterface(Exception):
    pass


SpecificRobotInterface: typing.Type[RobotInterface]

if os.environ.get("ROBOT_INTERFACE", "prod") == "prod":
    from computer_vision.robot_interface.prod_mode import (
        ProdMode as SpecificRobotInterface,
    )
elif os.environ.get("ROBOT_INTERFACE", "prod") == "dev":
    from computer_vision.robot_interface.dev_mode import (
        DevMode as SpecificRobotInterface,
    )
else:
    raise InvalidRobotInterface(os.environ.get("ROBOT_INTERFACE", "prod"))


def main() -> None:

    cameras: typing.List[cv2.VideoCapture] = get_available_cameras()

    for camera in cameras:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not os.environ.get("ALLOW_NO_CAMERAS", False) and len(cameras) < 2:
        raise Exception("You need at least two cameras to run this code.")
    elif os.environ.get("ALLOW_NO_CAMERAS", False) and len(cameras) < 2:
        while len(cameras) < 2:
            cameras.append(FakeCapture("./computer_vision/assets/image_example.jpg"))

    robot_interface = SpecificRobotInterface()

    robot_interface.start_interface()

    switch_cameras = robot_interface.should_switch_cameras()

    ball_camera, target_camera = set_cameras(cameras, switch_cameras)

    (
        target_hsv_range,
        red_ball_hsv_range,
        blue_ball_hsv_range,
        target_distance_parameters,
        ball_distance_parameters,
        target_contour_filter_parameters,
    ) = calibrate(robot_interface, ball_camera, target_camera)

    target_stream_resolution = robot_interface.get_target_stream_resolution()
    ball_stream_resolution = robot_interface.get_ball_stream_resolution()

    while True:

        working = robot_interface.refresh_interface()
        if not working:
            break

        switch_cameras = robot_interface.should_switch_cameras()

        ball_camera, target_camera = set_cameras(cameras, switch_cameras)

        is_calibrating = robot_interface.is_calibrating_all_cameras()

        if is_calibrating:
            (
                target_hsv_range,
                red_ball_hsv_range,
                blue_ball_hsv_range,
                target_distance_parameters,
                ball_distance_parameters,
                target_contour_filter_parameters,
            ) = calibrate(robot_interface, ball_camera, target_camera)

        target_camera_found, target_frame = target_camera.read()
        ball_camera_found, ball_frame = ball_camera.read()

        if not target_camera_found or not ball_camera_found:
            continue

        ball_color = robot_interface.get_ball_color()

        ball_hsv_range = {"red": red_ball_hsv_range, "blue": blue_ball_hsv_range}[
            ball_color
        ]

        target_response = process_target_image(
            target_frame, target_hsv_range, target_contour_filter_parameters
        )
        ball_response = process_ball_image(ball_frame, ball_hsv_range)

        robot_interface.send_if_target_was_found(target_response.found)
        robot_interface.send_if_ball_was_found(ball_response.found)

        # making calculations and sending results

        if target_response.found:
            target_distance = calculate_target_distance(
                target_response.target_position.y,
                target_distance_parameters,
            )
            target_angle = calculate_horizontal_angle(
                target_response.target_position.x,
                target_frame.shape[1],
                camera_offset=0,
            )
            launcher_angle = calculate_launcher_angle(target_distance)
            robot_interface.send_target_angle(target_angle)
            robot_interface.send_target_distance(target_distance)
            robot_interface.send_launcher_angle(launcher_angle)

        if ball_response.found:
            ball_distance = calculate_ball_distance(
                ball_response.ball_diameter, ball_distance_parameters
            )
            ball_angle = calculate_horizontal_angle(
                ball_response.ball_position.x,
                ball_frame.shape[1],
                camera_offset=0,
            )
            robot_interface.send_ball_angle(ball_angle)
            robot_interface.send_ball_distance(ball_distance)

        # feeding image streams

        robot_interface.send_target_camera_frame(
            resize(target_frame, target_stream_resolution)
        )
        robot_interface.send_ball_camera_frame(
            resize(ball_frame, ball_stream_resolution)
        )
        robot_interface.send_target_binary_frame(
            resize(target_response.binary_image, target_stream_resolution)
        )
        robot_interface.send_ball_binary_frame(
            resize(ball_response.binary_image, ball_stream_resolution)
        )
        robot_interface.send_frame_with_target(
            resize(target_response.image_with_target, target_stream_resolution)
        )
        robot_interface.send_frame_with_ball(
            resize(ball_response.image_with_ball, ball_stream_resolution)
        )

    robot_interface.stop_interface()


def set_cameras(
    cameras: typing.List[cv2.VideoCapture], switch_cameras: bool
) -> typing.Tuple[cv2.VideoCapture, cv2.VideoCapture]:
    if switch_cameras:
        ball_camera, target_camera = cameras
    else:
        target_camera, ball_camera = cameras
    return ball_camera, target_camera


def calibrate(
    robot_interface: RobotInterface,
    ball_camera: cv2.VideoCapture,
    target_camera: cv2.VideoCapture,
) -> typing.Tuple[
    HsvRange,
    HsvRange,
    HsvRange,
    TargetDistanceParameters,
    BallDistanceParameters,
    TargetContourFilterParameters,
]:
    target_hsv_range = robot_interface.get_target_hsv_range()
    red_ball_hsv_range = robot_interface.get_red_ball_hsv_range()
    blue_ball_hsv_range = robot_interface.get_blue_ball_hsv_range()
    target_brightness = robot_interface.get_target_camera_brightness()
    ball_brightness = robot_interface.get_ball_camera_brightness()
    target_exposure = robot_interface.get_target_camera_exposure()
    ball_exposure = robot_interface.get_ball_camera_exposure()
    set_brightness(target_camera, target_brightness)
    set_brightness(ball_camera, ball_brightness)
    set_exposure(target_camera, target_exposure)
    set_exposure(ball_camera, ball_exposure)
    target_distance_parameters = robot_interface.get_target_distance_parameters()
    ball_distance_parameters = robot_interface.get_ball_distance_parameters()
    target_contour_filter_parameters = (
        robot_interface.get_target_contour_filter_parameters()
    )
    camera_fps = robot_interface.get_fps()
    set_fps(target_camera, camera_fps)
    set_fps(ball_camera, camera_fps)
    return (
        target_hsv_range,
        red_ball_hsv_range,
        blue_ball_hsv_range,
        target_distance_parameters,
        ball_distance_parameters,
        target_contour_filter_parameters,
    )


if __name__ == "__main__":
    main()
