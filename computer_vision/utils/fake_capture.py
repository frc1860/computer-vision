import typing

import cv2
import numpy as np


class FakeCapture(cv2.VideoCapture):
    def __init__(self, file: str) -> None:
        self.image = cv2.imread(file)

    def read(self) -> typing.Tuple[bool, np.ndarray]:
        return True, self.image
