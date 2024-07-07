from __future__ import annotations

import os
from abc import ABC, abstractmethod
from logging import getLogger

from cv2.typing import MatLike
from numpy import float32
from numpy.typing import NDArray


class Detector(ABC):
    """
    Abstract class for detecting people in a video stream.
    """

    def __init__(self, *files: str) -> None:
        self._logger = getLogger(__name__)
        self._check_files(*files)

    @abstractmethod
    def detect_faces(self, frame: MatLike) -> NDArray[float32]:
        """
        Detect objects in a frame.

        Args:
            frame (MatLike): A frame from a video stream.

        Returns:
            NDArray[float32]: An array of bounding boxes consisting of the x0, y0, x1, y1 coordinates and the confidence level.
        """

        pass

    def _check_files(self, *files: str) -> None:
        """
        Check if all files exist.
        """
        for file in files:
            if not os.path.exists(file):
                self._logger.error(f"Error: '{file}' not found.")
                exit(1)
