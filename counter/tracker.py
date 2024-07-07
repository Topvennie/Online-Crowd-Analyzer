

from abc import ABC, abstractmethod
from logging import getLogger

from numpy import float32
from numpy.typing import NDArray


class Tracker(ABC):
    """
    Abstract class for tracking objects.
    """

    def __init__(self) -> None:
        self._logger = getLogger(__name__)

    @abstractmethod
    def update(self,
               rects: NDArray[float32]
               ) -> dict[int, tuple[int, int]]:
        """
        Update the tracker with new rectangles.

        Args:
            rects (NDArray): A list of rectangles representing the bounding boxes of objects.

        Returns:
            dict[int, NDArray]: A dictionary of object IDs and their corresponding centroids ((x, y) coordinates).
        """

        pass
