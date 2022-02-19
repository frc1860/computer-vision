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
    BallDistanceParameters,
    HsvRange,
    Position,
    Resolution,
    TargetDistanceParameters,
)

from .robot_interface import RobotInterface


@dataclass
class CameraInformation:
    is_available: bool = False
    stream_resolution: Resolution = Resolution()
    focal_length: float = 0
    brightness: float = 0


@dataclass
class LocalStorageInformation:
    calibrating_all_cameras: bool = False
    target_camera: CameraInformation = CameraInformation()
    ball_camera: CameraInformation = CameraInformation()
    target_hsv_range: HsvRange = HsvRange()
    red_ball_hsv_range: HsvRange = HsvRange()
    blue_ball_hsv_range: HsvRange = HsvRange()
    target_horizontal_angle: float = 0
    ball_horizontal_angle: float = 0
    target_position: Position = Position()
    ball_position: Position = Position()
    target_distance_parameters: TargetDistanceParameters = TargetDistanceParameters()
    ball_distance_parameters: BallDistanceParameters = BallDistanceParameters()
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

        block_1 = [
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

        block_21 = [
            [sg.Text("Target Camera")],
            [
                sg.Text("H min: "),
                sg.Slider(
                    key="target_h_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_hsv_range.hue.min,
                ),
            ],
            [
                sg.Text("H max: "),
                sg.Slider(
                    key="target_h_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_hsv_range.hue.max,
                ),
            ],
            [
                sg.Text("S min: "),
                sg.Slider(
                    key="target_s_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_hsv_range.saturation.min,
                ),
            ],
            [
                sg.Text("S max: "),
                sg.Slider(
                    key="target_s_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_hsv_range.saturation.max,
                ),
            ],
            [
                sg.Text("V min: "),
                sg.Slider(
                    key="target_v_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_hsv_range.value.min,
                ),
            ],
            [
                sg.Text("V max: "),
                sg.Slider(
                    key="target_v_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.target_hsv_range.value.max,
                ),
            ],
        ]

        block_22 = [
            [sg.Image(filename="", key="target_image")],
            [sg.Image(filename="", key="target_binary_image")],
            [sg.Image(filename="", key="image_with_target")],
        ]

        block_23 = [
            [sg.Text("Ball Camera")],
            [
                sg.Text("H min: "),
                sg.Slider(
                    key="red_ball_h_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.red_ball_hsv_range.hue.min,
                ),
            ],
            [
                sg.Text("H max: "),
                sg.Slider(
                    key="red_ball_h_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.red_ball_hsv_range.hue.max,
                ),
            ],
            [
                sg.Text("S min: "),
                sg.Slider(
                    key="red_ball_s_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.red_ball_hsv_range.saturation.min,
                ),
            ],
            [
                sg.Text("S max: "),
                sg.Slider(
                    key="red_ball_s_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.red_ball_hsv_range.saturation.max,
                ),
            ],
            [
                sg.Text("V min: "),
                sg.Slider(
                    key="red_ball_v_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.red_ball_hsv_range.value.min,
                ),
            ],
            [
                sg.Text("V max: "),
                sg.Slider(
                    key="red_ball_v_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.red_ball_hsv_range.value.max,
                ),
            ],
        ]

        block_24 = [
            [sg.Text("")],
            [
                sg.Text("H min: "),
                sg.Slider(
                    key="blue_ball_h_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.blue_ball_hsv_range.hue.min,
                ),
            ],
            [
                sg.Text("H max: "),
                sg.Slider(
                    key="blue_ball_h_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.blue_ball_hsv_range.hue.max,
                ),
            ],
            [
                sg.Text("S min: "),
                sg.Slider(
                    key="blue_ball_s_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.blue_ball_hsv_range.saturation.min,
                ),
            ],
            [
                sg.Text("S max: "),
                sg.Slider(
                    key="blue_ball_s_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.blue_ball_hsv_range.saturation.max,
                ),
            ],
            [
                sg.Text("V min: "),
                sg.Slider(
                    key="blue_ball_v_min",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.blue_ball_hsv_range.value.min,
                ),
            ],
            [
                sg.Text("V max: "),
                sg.Slider(
                    key="blue_ball_v_max",
                    orientation="horizontal",
                    resolution=0.1,
                    range=(0, 256),
                    default_value=data.blue_ball_hsv_range.value.max,
                ),
            ],
        ]

        block_25 = [
            [sg.Image(filename="", key="ball_image")],
            [sg.Image(filename="", key="ball_binary_image")],
            [sg.Image(filename="", key="image_with_ball")],
        ]

        block_2 = [
            sg.Column(block_21),
            sg.Column(block_22),
            sg.Column(block_23),
            sg.Column(block_24),
            sg.Column(block_25),
        ]

        block_3 = [
            [sg.Text("Target distance parameters")],
            [
                sg.Text("a: "),
                sg.InputText(key="target_distance_parameters_a"),
                sg.Text("b: "),
                sg.InputText(key="target_distance_parameters_b"),
                sg.Text("c: "),
                sg.InputText(key="target_distance_parameters_c"),
                sg.Button(
                    "SEND", button_color="green", key="send_target_distance_parameters"
                ),
            ],
        ]

        block_4 = [
            [sg.Text("Ball distance parameters")],
            [
                sg.Text("focal length: "),
                sg.InputText(key="ball_distance_parameters_focal_length"),
                sg.Text("ball diameter: "),
                sg.InputText(key="ball_distance_parameters_ball_diameter"),
                sg.Button(
                    "SEND", button_color="green", key="send_ball_distance_parameters"
                ),
            ],
        ]

        block_5 = [
            [sg.Text("Focal lengths")],
            [
                sg.Text("Target camera: "),
                sg.InputText(key="target_camera_focal_length"),
                sg.Text("Ball camera: "),
                sg.InputText(key="ball_camera_focal_length"),
                sg.Button("SEND", button_color="green", key="send_focal_lengths"),
            ],
        ]

        return window_name, [block_1, block_2, block_3, block_4, block_5]

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
        elif event == "calibration_switch":
            all_data.calibrating_all_cameras = not all_data.calibrating_all_cameras
            calibration_status = "ON" if all_data.calibrating_all_cameras else "OFF"
            calibration_button_color = (
                "green" if all_data.calibrating_all_cameras else "red"
            )
            self.window.Element("calibration_switch").Update(
                calibration_status, button_color=calibration_button_color
            )
        elif event == "switch_cameras_switch":
            all_data.switch_cameras = not all_data.switch_cameras
            switch_cameras_status = "ON" if all_data.switch_cameras else "OFF"
            switch_cameras_button_color = "green" if all_data.switch_cameras else "red"
            self.window.Element("switch_cameras_switch").Update(
                switch_cameras_status, button_color=switch_cameras_button_color
            )
        elif event == "send_target_distance_parameters":
            try:
                a = float(values["target_distance_parameters_a"])
                b = float(values["target_distance_parameters_b"])
                c = float(values["target_distance_parameters_c"])
                all_data.target_distance_parameters.a = a
                all_data.target_distance_parameters.b = b
                all_data.target_distance_parameters.c = c
            except Exception:
                pass
        elif event == "send_ball_distance_parameters":
            try:
                focal_length = float(values["ball_distance_parameters_focal_length"])
                ball_diameter = float(values["ball_distance_parameters_ball_diameter"])
                all_data.ball_distance_parameters.focal_length = focal_length
                all_data.ball_distance_parameters.ball_diameter = ball_diameter
            except Exception:
                pass
        elif event == "send_focal_lengths":
            try:
                target_camera_focal_length = float(values["target_camera_focal_length"])
                ball_camera_focal_length = float(values["ball_camera_focal_length"])
                all_data.target_camera.focal_length = target_camera_focal_length
                all_data.ball_camera.focal_length = ball_camera_focal_length
            except Exception:
                pass
        else:
            all_data.target_hsv_range.hue.min = values["target_h_min"]
            all_data.target_hsv_range.hue.max = values["target_h_max"]
            all_data.target_hsv_range.saturation.min = values["target_s_min"]
            all_data.target_hsv_range.saturation.max = values["target_s_max"]
            all_data.target_hsv_range.value.min = values["target_v_min"]
            all_data.target_hsv_range.value.max = values["target_v_max"]
            all_data.red_ball_hsv_range.hue.min = values["red_ball_h_min"]
            all_data.red_ball_hsv_range.hue.max = values["red_ball_h_max"]
            all_data.red_ball_hsv_range.saturation.min = values["red_ball_s_min"]
            all_data.red_ball_hsv_range.saturation.max = values["red_ball_s_max"]
            all_data.red_ball_hsv_range.value.min = values["red_ball_v_min"]
            all_data.red_ball_hsv_range.value.max = values["red_ball_v_max"]
            all_data.blue_ball_hsv_range.hue.min = values["blue_ball_h_min"]
            all_data.blue_ball_hsv_range.hue.max = values["blue_ball_h_max"]
            all_data.blue_ball_hsv_range.saturation.min = values["blue_ball_s_min"]
            all_data.blue_ball_hsv_range.saturation.max = values["blue_ball_s_max"]
            all_data.blue_ball_hsv_range.value.min = values["blue_ball_v_min"]
            all_data.blue_ball_hsv_range.value.max = values["blue_ball_v_max"]

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
        resized_frame = imutils.resize(frame, height=100)
        self.target_camera_last_frame = cv2.imencode(".png", resized_frame)[1].tobytes()

    def send_ball_camera_frame(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=100)
        self.ball_camera_last_frame = cv2.imencode(".png", resized_frame)[1].tobytes()

    def send_target_binary_frame(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=100)
        self.target_binary_last_frame = cv2.imencode(".png", resized_frame)[1].tobytes()

    def send_ball_binary_frame(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=100)
        self.ball_binary_last_frame = cv2.imencode(".png", resized_frame)[1].tobytes()

    def send_frame_with_target(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=100)
        self.last_frame_with_target = cv2.imencode(".png", resized_frame)[1].tobytes()

    def send_frame_with_ball(self, frame: np.ndarray) -> None:
        resized_frame = imutils.resize(frame, height=100)
        self.last_frame_with_ball = cv2.imencode(".png", resized_frame)[1].tobytes()

    def get_target_hsv_range(self) -> HsvRange:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.target_hsv_range

    def get_red_ball_hsv_range(self) -> HsvRange:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.red_ball_hsv_range

    def get_blue_ball_hsv_range(self) -> HsvRange:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.blue_ball_hsv_range

    def get_target_distance_parameters(self) -> TargetDistanceParameters:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.target_distance_parameters

    def get_ball_distance_parameters(self) -> BallDistanceParameters:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.ball_distance_parameters

    def get_target_camera_focal_length(self) -> float:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.target_camera.focal_length

    def get_ball_camera_focal_length(self) -> float:
        all_data = DevMode.extract_file(self.filepath)
        return all_data.ball_camera.focal_length

    def get_ball_color(self) -> typing.Literal["red", "blue"]:
        # TODO: Implement get_ball_color in DevMode
        return "red"

    def send_target_angle(self, angle: float) -> None:
        # TODO: Implement send_target_angle in DevMode
        pass

    def send_ball_angle(self, angle: float) -> None:
        # TODO: Implement send_ball_angle in DevMode
        pass

    def send_target_distance(self, distance: float) -> None:
        # TODO: Implement send_target_distance in DevMode
        pass

    def send_ball_distance(self, distance: float) -> None:
        # TODO: Implement send_ball_distance in DevMode
        pass

    def send_if_target_was_found(self, found: bool) -> None:
        # TODO: Implement send_if_target_was_found in DevMode
        pass

    def send_if_ball_was_found(self, found: bool) -> None:
        # TODO: Implement send_if_ball_was_found in DevMode
        pass

    def send_launcher_angle(self, angle: float) -> None:
        # TODO: Implement send_launcher_angle in DevMode
        pass
