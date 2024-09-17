import typing

import cv2
import numpy as np


class FakeCapture(cv2.VideoCapture):
    def __init__(self, file: str) -> None:
        self.file = file

    def read(self) -> typing.Tuple[bool, np.ndarray]:
        return True, cv2.imread(self.file)

    def set(self, _: int, __: float) -> None:
        pass
