import json
import logging
import os
import time
import typing
from dataclasses import dataclass

import cv2
import imutils
import numpy as np
import PySimpleGUI as sg

from computer_vision.utils.dataclass_to_json import extract_dataclass, load_dataclass
from computer_vision.utils.internal_types import (
    DistanceCalculationParameters,
    HsvRange,
    Position,
    Resolution,
)

from .robot_interface import RobotInterface


@dataclass
class CameraInformation:
    is_available: bool = False
    stream_resolution: Resolution = Resolution()
    hsv_range: HsvRange = HsvRange()
    focal_length: float = 0
    brightness: float = 0


@dataclass
class LocalStorageInformation:
    calibrating_all_cameras: bool = False
    target_camera: CameraInformation = CameraInformation()
    ball_camera: CameraInformation = CameraInformation()
    target_horizontal_angle: float = 0
    ball_horizontal_angle: float = 0
    target_position: Position = Position()
    ball_position: Position = Position()
    launcher_angle: float = 0
    distance_calculation_parameters: DistanceCalculationParameters = (
        DistanceCalculationParameters()
    )
    switch_cameras: bool = False


class DevMode(RobotInterface):

    FILENAME = "local_storage.json"

    class InvalidDirectory(Exception):
        pass

    class FailedToExtractJson(Exception):
        MAX_RETRIES = 5

    class FailedToLoadDataclass(Exception):
        MAX_RETRIES = 5

    def __init__(self, directory: str = "./") -> None:

        directory_exists = os.path.isdir(directory)
        if not directory_exists:
            raise DevMode.InvalidDirectory(directory)

        filepath = os.path.join(directory, DevMode.FILENAME)

        self.filepath = filepath

        if DevMode.file_is_empty(filepath):
            DevMode.load_file(filepath, LocalStorageInformation())

        self.target_camera_last_frame: typing.Optional[np.ndarray] = None
        self.ball_camera_last_frame: typing.Optional[np.ndarray] = None
        self.target_binary_last_frame: typing.Optional[np.ndarray] = None
        self.ball_binary_last_frame: typing.Optional[np.ndarray] = None
        self.last_frame_with_target: typing.Optional[np.ndarray] = None
        self.last_frame_with_ball: typing.Optional[np.ndarray] = None

    @staticmethod
    def build_layout(
        data: LocalStorageInformation,
    ) -> typing.Tuple[str, typing.List[typing.List[sg.Element]]]:

        window_name = "Local Interface"

        calibration_status = "ON" if data.calibrating_all_cameras else "OFF"
        calibration_button_color = "green" if data.calibrating_all_cameras else "red"

        switch_cameras_status = "ON" if data.switch_cameras else "OFF"
        switch_cameras_button_color = "green" if data.switch_cameras else "red"

        first_row = [
            sg.Text("Calibration: "),
            sg.Button(
                calibration_status,
                button_color=calibration_button_color,
                key="calibration_switch",
            ),
            sg.Text("Switch cameras: "),
            sg.Button(
                switch_cameras_status,
                button_color=switch_cameras_button_color,
                key="switch_cameras_switch",
            ),
        ]

        first_column = [
            [sg.Text("Target Camera")],
            [
                sg.Text("H min: "),
                sg.Slider(
                    key="target_h_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_camera.hsv_range.hue.min,
                ),
            ],
            [
                sg.Text("H max: "),
                sg.Slider(
                    key="target_h_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_camera.hsv_range.hue.max,
                ),
            ],
            [
                sg.Text("S min: "),
                sg.Slider(
                    key="target_s_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_camera.hsv_range.saturation.min,
                ),
            ],
            [
                sg.Text("S max: "),
                sg.Slider(
                    key="target_s_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_camera.hsv_range.saturation.max,
                ),
            ],
            [
                sg.Text("V min: "),
                sg.Slider(
                    key="target_v_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_camera.hsv_range.value.min,
                ),
            ],
            [
                sg.Text("V max: "),
                sg.Slider(
                    key="target_v_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_camera.hsv_range.value.max,
                ),
            ],
        ]

        second_column = [
            [sg.Image(filename="", key="target_image")],
            [sg.Image(filename="", key="target_binary_image")],
            [sg.Image(filename="", key="image_with_target")],
        ]

        third_column = [
            [sg.Text("Ball Camera")],
            [
                sg.Text("H min: "),
                sg.Slider(
                    key="ball_h_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.ball_camera.hsv_range.hue.min,
                ),
            ],
            [
                sg.Text("H max: "),
                sg.Slider(
                    key="ball_h_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.ball_camera.hsv_range.hue.max,
                ),
            ],
            [
                sg.Text("S min: "),
                sg.Slider(
                    key="ball_s_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.ball_camera.hsv_range.saturation.min,
                ),
            ],
            [
                sg.Text("S max: "),
                sg.Slider(
                    key="ball_s_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.ball_camera.hsv_range.saturation.max,
                ),
            ],
            [
                sg.Text("V min: "),
                sg.Slider(
                    key="ball_v_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.ball_camera.hsv_range.value.min,
                ),
            ],
            [
                sg.Text("V max: "),
                sg.Slider(
                    key="ball_v_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.ball_camera.hsv_range.value.max,
                ),
            ],
        ]

        fourth_column = [
            [sg.Image(filename="", key="ball_image")],
            [sg.Image(filename="", key="ball_binary_image")],
            [sg.Image(filename="", key="image_with_ball")],
        ]

        second_row = [
            sg.Column(first_column),
            sg.Column(second_column),
            sg.Column(third_column),
            sg.Column(fourth_column),
        ]

        return window_name, [first_row, second_row]

    @staticmethod
    def file_is_empty(filepath: str) -> bool:
        with open(filepath, "r") as file:
            content = file.read()
            if content is None or content.strip() == "":
                return True
            else:
                return False

    @staticmethod
    def extract_file(filepath: str) -> LocalStorageInformation:
        for _ in range(DevMode.FailedToExtractJson.MAX_RETRIES):
            try:
                with open(filepath, "r") as file:
                    data = typing.cast(
                        LocalStorageInformation,
                        extract_dataclass(file.read(), LocalStorageInformation),
                    )
                    return data
            except json.decoder.JSONDecodeError:
                logging.warning("JSON decoder failed, retring...")
                time.sleep(1)
        with open(filepath, "r") as file:
            raise DevMode.FailedToExtractJson(file.read())

    @staticmethod
    def load_file(filepath: str, data: LocalStorageInformation) -> None:
        for _ in range(DevMode.FailedToLoadDataclass.MAX_RETRIES):
            try:
                with open(filepath, "w") as file:
                    file.write(load_dataclass(data))
                    return None
            except Exception as e:
                logging.warning(f"JSON decoder failed, retring... (error: {e})")
                time.sleep(1)
        raise DevMode.FailedToLoadDataclass(data)

    def start_interface(self) -> None:

        all_data = DevMode.extract_file(self.filepath)
        name, layout = DevMode.build_layout(all_data)
        self.window = sg.Window(name, layout)

    def refresh_interface(self) -> bool:

        all_data = DevMode.extract_file(self.filepath)
        event, values = self.window.read(timeout=10)

        if event == sg.WIN_CLOSED:
            return False
        if event == "calibration_switch":
            all_data.calibrating_all_cameras = not all_data.calibrating_all_cameras
            calibration_status = "ON" if all_data.calibrating_all_cameras else "OFF"
            calibration_button_color = (
                "green" if all_data.calibrating_all_cameras else "red"
            )
            self.window.Element("calibration_switch").Update(
                calibration_status, button_color=calibration_button_color
            )
        if event == "switch_cameras_switch":
            all_data.switch_cameras = not all_data.switch_cameras
            switch_cameras_status = "ON" if all_data.switch_cameras else "OFF"
            switch_cameras_button_color = "green" if all_data.switch_cameras else "red"
            self.window.Element("switch_cameras_switch").Update(
                switch_cameras_status, button_color=switch_cameras_button_color
            )

        else:
            all_data.target_camera.hsv_range.hue.min = values["target_h_min"]
            all_data.target_camera.hsv_range.hue.max = values["target_h_max"]
            all_data.target_camera.hsv_range.saturation.min = values["target_s_min"]
            all_data.target_camera.hsv_range.saturation.max = values["target_s_max"]
            all_data.target_camera.hsv_range.value.min = values["target_v_min"]
            all_data.target_camera.hsv_range.value.max = values["target_v_max"]
            all_data.ball_camera.hsv_range.hue.min = values["ball_h_min"]
            all_data.ball_camera.hsv_range.hue.max = values["ball_h_max"]
            all_data.ball_camera.hsv_range.saturation.min = values["ball_s_min"]
            all_data.ball_camera.hsv_range.saturation.max = values["ball_s_max"]
            all_data.ball_camera.hsv_range.value.min = values["ball_v_min"]
            all_data.ball_camera.hsv_range.value.max = values["ball_v_max"]

            self.window.Element("target_image").update(
                data=self.target_camera_last_frame
            )
            self.window.Element("ball_image").update(data=self.ball_camera_last_frame)

            self.window.Element("target_binary_image").update(
                data=self.target_binary_last_frame
            )
            self.window.Element("ball_binary_image").update(
                data=self.ball_binary_last_frame
            )

            self.window.Element("image_with_target").update(
                data=self.last_frame_with_target
            )
            self.window.Element("image_with_ball").update(
                data=self.last_frame_with_ball
            )

        DevMode.load_file(self.filepath, all_data)

        return True

    def stop_interface(self) -> None:
        self.window.close()

    def should_switch_cameras(self) -> bool:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.switch_cameras

    def is_calibrating_all_cameras(self) -> bool:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.calibrating_all_cameras

    def get_target_stream_resolution(self) -> Resolution:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.target_camera.stream_resolution

    def get_ball_stream_resolution(self) -> Resolution:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.ball_camera.stream_resolution

    def send_target_camera_frame(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=150)
        self.target_camera_last_frame = cv2.imencode(".png", resized_frame)[1].tobytes()

    def send_ball_camera_frame(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=150)
        self.ball_camera_last_frame = cv2.imencode(".png", resized_frame)[1].tobytes()

    def send_target_binary_frame(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=150)
        self.target_binary_last_frame = cv2.imencode(".png", resized_frame)[1].tobytes()

    def send_ball_binary_frame(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=150)
        self.ball_binary_last_frame = cv2.imencode(".png", resized_frame)[1].tobytes()

    def send_frame_with_target(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=150)
        self.last_frame_with_target = cv2.imencode(".png", resized_frame)[1].tobytes()

    def send_frame_with_ball(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=150)
        self.last_frame_with_ball = cv2.imencode(".png", resized_frame)[1].tobytes()

    def get_target_camera_hsv_range(self) -> HsvRange:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.target_camera.hsv_range

    def get_ball_camera_hsv_range(self) -> HsvRange:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.ball_camera.hsv_range
