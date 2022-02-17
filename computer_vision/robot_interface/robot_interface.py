from abc import ABCMeta, abstractmethod

import numpy as np

from computer_vision.utils.internal_types import HsvRange, Resolution


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
    def get_target_camera_hsv_range(self) -> HsvRange:
        pass

    @abstractmethod
    def get_ball_camera_hsv_range(self) -> HsvRange:
        pass
