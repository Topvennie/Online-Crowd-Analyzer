import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from counter.detector import Detector


class HOG(Detector):
    """
    Detector using the Histogram of Oriented Gradients (HOG) method.

    HOG, Histogram of Oriented Gradients, is a feature descriptor.
    It uses the distribution of intensity gradients and edge directions to detect objects.
    """

    def __init__(self, confidence: float) -> None:
        super().__init__()

        self._confidence = confidence
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # type: ignore

    def detect_faces(self, frame: MatLike) -> NDArray[np.float32]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = self._hog.detectMultiScale(gray, winStride=(8, 8), padding=(0, 0))
        boxes = np.array([[x, y, x + w, y + h, weights[index]]
                         for (index, (x, y, w, h)) in enumerate(boxes) if weights[index] > self._confidence])

        return boxes
