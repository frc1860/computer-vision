import os
import typing

import cv2

from computer_vision.robot_interface.dev_mode import DevMode
from computer_vision.robot_interface.prod_mode import ProdMode
from computer_vision.robot_interface.robot_interface import RobotInterface
from computer_vision.utils.fake_capture import FakeCapture
from computer_vision.utils.image_processing import (
    get_available_cameras,
    get_frame,
    process_ball_image,
    process_target_image,
    resize,
)


class InvalidRobotInterface(Exception):
    pass


def main() -> None:

    cameras: typing.List[cv2.VideoCapture] = get_available_cameras()

    if not os.environ.get("ALLOW_NO_CAMERAS", False) and len(cameras) < 2:
        raise Exception("You need at least two cameras to run this code.")
    elif os.environ.get("ALLOW_NO_CAMERAS", False) and len(cameras) < 2:
        while len(cameras) < 2:
            cameras.append(FakeCapture("./computer_vision/assets/image_example.jpg"))

    robot_interfaces: typing.Dict[str, typing.Type[RobotInterface]] = {
        "dev": DevMode,
        "prod": ProdMode,
    }

    chosen_robot_interface = os.environ.get("ROBOT_INTERFACE", "prod")

    try:
        robot_interface = robot_interfaces[chosen_robot_interface]()
    except KeyError:
        raise InvalidRobotInterface(chosen_robot_interface)

    robot_interface.start_interface()

    target_camera_hsv_range = robot_interface.get_target_camera_hsv_range()
    ball_camera_hsv_range = robot_interface.get_ball_camera_hsv_range()

    target_stream_resolution = robot_interface.get_target_stream_resolution()
    ball_stream_resolution = robot_interface.get_ball_stream_resolution()

    while True:

        working = robot_interface.refresh_interface()
        if not working:
            break

        target_camera, ball_camera = cameras

        is_calibrating = robot_interface.is_calibrating_all_cameras()

        if is_calibrating:
            target_camera_hsv_range = robot_interface.get_target_camera_hsv_range()
            ball_camera_hsv_range = robot_interface.get_ball_camera_hsv_range()

        target_frame = get_frame(target_camera)
        ball_frame = get_frame(ball_camera)

        target_response = process_target_image(target_frame, target_camera_hsv_range)
        ball_response = process_ball_image(ball_frame, ball_camera_hsv_range)

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


if __name__ == "__main__":
    main()
