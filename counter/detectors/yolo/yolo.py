import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine import results

from counter.detector import Detector

MODEL_PATH = "counter/detectors/yolo/"

CLASSES_INDEX = 0


class Yolo(Detector):

    def __init__(self, confidence, model):
        """
        Detector using the YOLO method.

        YOLO, You Only Look Once, is a real-time object detection system.
        It applies a single neural network to the full image, making predictions directly from the full image.

        Different models can be used, such as YOLOv3, YOLOv4, YOLOv5, etc.
        """

        super().__init__()

        self._confidence = confidence
        self._model = YOLO(MODEL_PATH + model, verbose=False)

    def detect_faces(self, frame: MatLike) -> NDArray[np.float32]:
        detections = self._model.predict(frame, classes=[CLASSES_INDEX], conf=self._confidence, verbose=False)

        return process_detections(detections[0])


def process_detections(detections: results.Results) -> NDArray[np.float32]:
    """
    Converts the detections to a numpy array.
    """

    processed = np.empty((len(detections), 5), dtype=np.float32)  # type: ignore

    detections = detections.cpu()

    processed[:, 1:] = detections.boxes.xyxy  # type: ignore
    processed[:, 0] = detections.boxes.conf  # type: ignore

    return processed
