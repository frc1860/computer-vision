import typing
from abc import ABCMeta, abstractmethod

import numpy as np

from computer_vision.utils.internal_types import (
    BallDistanceParameters,
    HsvRange,
    Resolution,
    TargetDistanceParameters,
)


class RobotInterface(metaclass=ABCMeta):
    @abstractmethod
    def start_interface(self) -> None:
        pass

    @abstractmethod
    def refresh_interface(self) -> bool:
        pass

    @abstractmethod
    def stop_interface(self) -> None:
        pass

    @abstractmethod
    def should_switch_cameras(self) -> bool:
        pass

    @abstractmethod
    def is_calibrating_all_cameras(self) -> bool:
        pass

    @abstractmethod
    def get_target_stream_resolution(self) -> Resolution:
        pass

    @abstractmethod
    def get_ball_stream_resolution(self) -> Resolution:
        pass

    @abstractmethod
    def send_target_camera_frame(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def send_ball_camera_frame(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def send_target_binary_frame(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def send_ball_binary_frame(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def send_frame_with_target(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def send_frame_with_ball(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_target_hsv_range(self) -> HsvRange:
        pass

    @abstractmethod
    def get_red_ball_hsv_range(self) -> HsvRange:
        pass

    @abstractmethod
    def get_blue_ball_hsv_range(self) -> HsvRange:
        pass

    @abstractmethod
    def get_target_distance_parameters(self) -> TargetDistanceParameters:
        pass

    @abstractmethod
    def get_ball_distance_parameters(self) -> BallDistanceParameters:
        pass

    @abstractmethod
    def get_target_camera_focal_length(self) -> float:
        pass

    @abstractmethod
    def get_ball_camera_focal_length(self) -> float:
        pass

    @abstractmethod
    def get_ball_color(self) -> typing.Literal["red", "blue"]:
        pass

    @abstractmethod
    def send_target_angle(self, angle: float) -> None:
        pass

    @abstractmethod
    def send_ball_angle(self, angle: float) -> None:
        pass

    @abstractmethod
    def send_target_distance(self, distance: float) -> None:
        pass

    @abstractmethod
    def send_ball_distance(self, distance: float) -> None:
        pass

    @abstractmethod
    def send_if_target_was_found(self, found: bool) -> None:
        pass

    @abstractmethod
    def send_if_ball_was_found(self, found: bool) -> None:
        pass

    @abstractmethod
    def send_launcher_angle(self, angle: float) -> None:
        pass
