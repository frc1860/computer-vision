import typing
from dataclasses import dataclass

import numpy as np
from cscore import CameraServer
from networktables import NetworkTablesInstance

from computer_vision.utils.internal_types import (
    BallDistanceParameters,
    HsvRange,
    Range,
    Resolution,
    TargetContourFilterParameters,
    TargetDistanceParameters,
)

from .robot_interface import RobotInterface


@dataclass
class TargetContourFilterParametersEntries:
    min_area: typing.Any
    min_perimeter: typing.Any
    min_width: typing.Any
    max_width: typing.Any
    min_height: typing.Any
    max_height: typing.Any
    min_solidity: typing.Any
    max_solidity: typing.Any
    min_vertex_count: typing.Any
    max_vertex_count: typing.Any
    min_ratio: typing.Any
    max_ratio: typing.Any


@dataclass
class TargetEntries:
    resolution: typing.Any
    focal_length: typing.Any
    hsv_range: typing.Any
    distance_parameters: typing.Any
    angle: typing.Any
    distance: typing.Any
    found: typing.Any
    brightness: typing.Any
    exposure: typing.Any
    contour_filter_parameters: TargetContourFilterParametersEntries


@dataclass
class BallEntries:
    resolution: typing.Any
    focal_length: typing.Any
    ball_color: typing.Any
    red_hsv_range: typing.Any
    blue_hsv_range: typing.Any
    distance_parameters: typing.Any
    red_angle: typing.Any
    blue_angle: typing.Any
    red_distance: typing.Any
    blue_distance: typing.Any
    red_found: typing.Any
    blue_found: typing.Any
    brightness: typing.Any
    exposure: typing.Any


@dataclass
class AllNetTableEntries:
    is_calibrating: typing.Any
    fps: typing.Any
    switch_cameras: typing.Any
    target_entries: TargetEntries
    ball_entries: BallEntries
    launcher_angle: typing.Any


class ProdMode(RobotInterface):
    def __init__(self) -> None:

        netTable = NetworkTablesInstance.getDefault()
        netTable.startClientTeam(1860)

        # Initializing network tables variables
        self.allNetTableEntries = AllNetTableEntries(
            is_calibrating=netTable.getEntry("/IsCalibrating"),
            fps=netTable.getEntry("/Fps"),
            switch_cameras=netTable.getEntry("/SwitchCameras"),
            target_entries=TargetEntries(
                resolution=netTable.getEntry("/Target/Resolution"),
                focal_length=netTable.getEntry("/Target/FocalLength"),
                hsv_range=netTable.getEntry("/Target/HsvRange"),
                distance_parameters=netTable.getEntry("/Target/DistanceParameters"),
                angle=netTable.getEntry("/Target/Angle"),
                distance=netTable.getEntry("/Target/Distance"),
                found=netTable.getEntry("/Target/Found"),
                brightness=netTable.getEntry("/Target/Brightness"),
                exposure=netTable.getEntry("/Target/Exposure"),
                contour_filter_parameters=TargetContourFilterParametersEntries(
                    min_area=netTable.getEntry("/Target/MinArea"),
                    min_perimeter=netTable.getEntry("/Target/MinPerimeter"),
                    min_width=netTable.getEntry("/Target/MinWidth"),
                    max_width=netTable.getEntry("/Target/MaxWidth"),
                    min_height=netTable.getEntry("/Target/MinHeight"),
                    max_height=netTable.getEntry("/Target/MaxHeight"),
                    min_solidity=netTable.getEntry("/Target/MinSolidity"),
                    max_solidity=netTable.getEntry("/Target/MaxSolidity"),
                    min_vertex_count=netTable.getEntry("/Target/MinVertexCount"),
                    max_vertex_count=netTable.getEntry("/Target/MaxVertexCount"),
                    min_ratio=netTable.getEntry("/Target/MinRatio"),
                    max_ratio=netTable.getEntry("/Target/MaxRatio"),
                ),
            ),
            ball_entries=BallEntries(
                resolution=netTable.getEntry("/Ball/Resolution"),
                focal_length=netTable.getEntry("/Ball/FocalLength"),
                ball_color=netTable.getEntry("/Ball/Color"),
                red_hsv_range=netTable.getEntry("/Ball/RedHsvRange"),
                blue_hsv_range=netTable.getEntry("/Ball/BlueHsvRange"),
                distance_parameters=netTable.getEntry("/Ball/DistanceParameters"),
                red_angle=netTable.getEntry("/Ball/RedAngle"),
                blue_angle=netTable.getEntry("/Ball/BlueAngle"),
                red_distance=netTable.getEntry("/Ball/RedDistance"),
                blue_distance=netTable.getEntry("/Ball/BlueDistance"),
                red_found=netTable.getEntry("/Ball/RedFound"),
                blue_found=netTable.getEntry("/Ball/BlueFound"),
                brightness=netTable.getEntry("/Ball/Brightness"),
                exposure=netTable.getEntry("/Ball/Exposure"),
            ),
            launcher_angle=netTable.getEntry("/LauncherAngle"),
        )

        self.cs = CameraServer.getInstance()
        self.cs.enableLogging()

        target_resolution = self.get_target_stream_resolution()
        ball_resolution = self.get_ball_stream_resolution()

        self.target_camera_stream = self.create_stream(
            "TargetCamera", target_resolution
        )
        self.ball_camera_stream = self.create_stream("BallCamera", ball_resolution)
        self.target_binary_stream = self.create_stream(
            "TargetBinary", target_resolution
        )
        self.ball_binary_stream = self.create_stream("BallBinary", ball_resolution)
        self.image_with_target_stream = self.create_stream(
            "ImageWithTarget", target_resolution
        )
        self.image_with_ball_stream = self.create_stream(
            "ImageWithBall", ball_resolution
        )

    def create_stream(self, name: str, resolution: Resolution) -> typing.Any:
        return self.cs.putVideo(name, resolution.width, resolution.height)

    def start_interface(self) -> None:
        pass

    def refresh_interface(self) -> bool:
        return True

    def stop_interface(self) -> None:
        pass

    def get_fps(self) -> int:
        return self.allNetTableEntries.fps.getDouble(30)

    def should_switch_cameras(self) -> bool:
        return self.allNetTableEntries.switch_cameras.getBoolean(False)

    def is_calibrating_all_cameras(self) -> bool:
        return self.allNetTableEntries.is_calibrating.getBoolean(False)

    def get_target_stream_resolution(self) -> Resolution:
        (
            width,
            height,
        ) = self.allNetTableEntries.target_entries.resolution.getDoubleArray(
            [1280, 720]
        )
        return Resolution(width=width, height=height)

    def get_ball_stream_resolution(self) -> Resolution:
        width, height = self.allNetTableEntries.ball_entries.resolution.getDoubleArray(
            [1280, 720]
        )
        return Resolution(width=width, height=height)

    def send_target_camera_frame(self, frame: np.ndarray) -> None:
        self.target_camera_stream.putFrame(frame)

    def send_ball_camera_frame(self, frame: np.ndarray) -> None:
        self.ball_camera_stream.putFrame(frame)

    def send_target_binary_frame(self, frame: np.ndarray) -> None:
        self.target_binary_stream.putFrame(frame)

    def send_ball_binary_frame(self, frame: np.ndarray) -> None:
        self.ball_binary_stream.putFrame(frame)

    def send_frame_with_target(self, frame: np.ndarray) -> None:
        self.image_with_target_stream.putFrame(frame)

    def send_frame_with_ball(self, frame: np.ndarray) -> None:
        self.image_with_ball_stream.putFrame(frame)

    def get_target_hsv_range(self) -> HsvRange:
        (
            h_min,
            h_max,
            s_min,
            s_max,
            v_min,
            v_max,
        ) = self.allNetTableEntries.target_entries.hsv_range.getDoubleArray(
            [0, 256, 0, 256, 0, 256]
        )
        return HsvRange(
            hue=Range(min=h_min, max=h_max),
            saturation=Range(min=s_min, max=s_max),
            value=Range(min=v_min, max=v_max),
        )

    def get_target_contour_filter_parameters(self) -> TargetContourFilterParameters:
        min_area = self.allNetTableEntries.target_entries.contour_filter_parameters.min_area.getDouble(
            100
        )
        min_perimeter = self.allNetTableEntries.target_entries.contour_filter_parameters.min_perimeter.getDouble(
            0
        )
        min_width = self.allNetTableEntries.target_entries.contour_filter_parameters.min_width.getDouble(
            20
        )
        max_width = self.allNetTableEntries.target_entries.contour_filter_parameters.max_width.getDouble(
            400
        )
        min_height = self.allNetTableEntries.target_entries.contour_filter_parameters.min_height.getDouble(
            15
        )
        max_height = self.allNetTableEntries.target_entries.contour_filter_parameters.max_height.getDouble(
            250
        )
        min_solidity = self.allNetTableEntries.target_entries.contour_filter_parameters.min_solidity.getDouble(
            0
        )
        max_solidity = self.allNetTableEntries.target_entries.contour_filter_parameters.max_solidity.getDouble(
            100
        )
        min_vertex_count = self.allNetTableEntries.target_entries.contour_filter_parameters.min_vertex_count.getDouble(
            0
        )
        max_vertex_count = self.allNetTableEntries.target_entries.contour_filter_parameters.max_vertex_count.getDouble(
            10500
        )
        min_ratio = self.allNetTableEntries.target_entries.contour_filter_parameters.min_ratio.getDouble(
            0
        )
        max_ratio = self.allNetTableEntries.target_entries.contour_filter_parameters.max_ratio.getDouble(
            1000
        )
        return TargetContourFilterParameters(
            area=Range(min=min_area, max=100000),
            perimeter=Range(min=min_perimeter, max=100000),
            width=Range(min=min_width, max=max_width),
            height=Range(min=min_height, max=max_height),
            solidity=Range(min=min_solidity, max=max_solidity),
            vertex_count=Range(min=min_vertex_count, max=max_vertex_count),
            ratio=Range(min=min_ratio, max=max_ratio),
        )

    def get_red_ball_hsv_range(self) -> HsvRange:
        (
            h_min,
            h_max,
            s_min,
            s_max,
            v_min,
            v_max,
        ) = self.allNetTableEntries.ball_entries.red_hsv_range.getDoubleArray(
            [0, 256, 0, 256, 0, 256]
        )
        return HsvRange(
            hue=Range(min=h_min, max=h_max),
            saturation=Range(min=s_min, max=s_max),
            value=Range(min=v_min, max=v_max),
        )

    def get_blue_ball_hsv_range(self) -> HsvRange:
        (
            h_min,
            h_max,
            s_min,
            s_max,
            v_min,
            v_max,
        ) = self.allNetTableEntries.ball_entries.blue_hsv_range.getDoubleArray(
            [0, 256, 0, 256, 0, 256]
        )
        return HsvRange(
            hue=Range(min=h_min, max=h_max),
            saturation=Range(min=s_min, max=s_max),
            value=Range(min=v_min, max=v_max),
        )

    def get_target_distance_parameters(self) -> TargetDistanceParameters:
        (
            a,
            b,
            c,
        ) = self.allNetTableEntries.target_entries.distance_parameters.getDoubleArray(
            [1, 1, 1]
        )
        return TargetDistanceParameters(a=a, b=b, c=c)

    def get_ball_distance_parameters(self) -> BallDistanceParameters:
        (
            focal_length,
            ball_diameter,
        ) = self.allNetTableEntries.ball_entries.distance_parameters.getDoubleArray(
            [1, 1]
        )
        return BallDistanceParameters(
            focal_length=focal_length, ball_diameter=ball_diameter
        )

    def get_target_camera_focal_length(self) -> float:
        return self.allNetTableEntries.target_entries.focal_length.getDouble(1)

    def get_ball_camera_focal_length(self) -> float:
        return self.allNetTableEntries.ball_entries.focal_length.getDouble(1)

    def get_target_camera_brightness(self) -> float:
        return self.allNetTableEntries.target_entries.brightness.getDouble(0.5)

    def get_ball_camera_brightness(self) -> float:
        return self.allNetTableEntries.ball_entries.brightness.getDouble(0.5)

    def get_target_camera_exposure(self) -> float:
        return self.allNetTableEntries.target_entries.exposure.getDouble(0.01)

    def get_ball_camera_exposure(self) -> float:
        return self.allNetTableEntries.ball_entries.exposure.getDouble(0.01)

    def get_ball_color(self) -> str:
        color_number = self.allNetTableEntries.ball_entries.ball_color.getDouble(1)
        color = {1: "red", 2: "blue"}[int(color_number)]
        return color

    def send_target_angle(self, angle: float) -> None:
        self.allNetTableEntries.target_entries.angle.setDouble(angle)

    def send_ball_angle(self, angle: float) -> None:
        color = self.get_ball_color()
        red_multiplier, blue_multiplier = {"red": (1, 0), "blue": (0, 1)}[color]
        red_angle = angle * red_multiplier
        blue_angle = angle * blue_multiplier
        self.allNetTableEntries.ball_entries.red_angle.setDouble(red_angle)
        self.allNetTableEntries.ball_entries.blue_angle.setDouble(blue_angle)

    def send_target_distance(self, distance: float) -> None:
        self.allNetTableEntries.target_entries.distance.setDouble(distance)

    def send_ball_distance(self, distance: float) -> None:
        color = self.get_ball_color()
        red_multiplier, blue_multiplier = {"red": (1, 0), "blue": (0, 1)}[color]
        red_distance = distance * red_multiplier
        blue_distance = distance * blue_multiplier
        self.allNetTableEntries.ball_entries.red_distance.setDouble(red_distance)
        self.allNetTableEntries.ball_entries.blue_distance.setDouble(blue_distance)

    def send_if_target_was_found(self, found: bool) -> None:
        self.allNetTableEntries.target_entries.found.setBoolean(found)

    def send_if_ball_was_found(self, found: bool) -> None:
        color = self.get_ball_color()
        red_multiplier, blue_multiplier = {"red": (1, 0), "blue": (0, 1)}[color]
        red_found = bool(found * red_multiplier)
        blue_found = bool(found * blue_multiplier)
        self.allNetTableEntries.ball_entries.red_found.setBoolean(red_found)
        self.allNetTableEntries.ball_entries.blue_found.setBoolean(blue_found)

    def send_launcher_angle(self, angle: float) -> None:
        self.allNetTableEntries.launcher_angle.setDouble(angle)
