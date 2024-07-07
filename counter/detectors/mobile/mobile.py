import cv2
import imutils
import numpy as np
from cv2.typing import MatLike
from numba import jit
from numpy.typing import NDArray

from counter.detector import Detector

MODEL_PATH = "counter/detectors/mobile/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "counter/detectors/mobile/MobileNetSSD_deploy.prototxt"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


class MobileNetSSD(Detector):
    """
    Detector using the MobileNet SSD model.

    MobileNet is a type of convolutional neural network (CNN) designed for mobile and embedded vision applications.
    A Single Shot Detector (SSD) uses a single deep neural network to predict the bounding boxes of objects in an image,
    sacrificing some accuracy for speed.
    """

    def __init__(self, confidence: float) -> None:
        super().__init__(MODEL_PATH, PROTOTXT)

        self._confidence = confidence
        self._model = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_PATH)
        # Will use CUDA if available
        self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self._W = None
        self._H = None

    def detect_faces(self, frame: MatLike) -> NDArray[np.float32]:
        if self._W is None or self._H is None:
            self._H, self._W = frame.shape[:2]
        frame = imutils.resize(frame, width=300)
        (W, H) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), [127.5])
        self._model.setInput(blob)
        detections: NDArray[np.float32] = self._model.forward()  # type: ignore

        return filter_detections(
            detections,
            self._confidence,
            CLASSES.index("person"),
            self._W,
            self._H
        )


@jit(nopython=True, fastmath=True, cache=True)
def filter_detections(detections: NDArray[np.float32], confidence: float, class_id: int, W: int, H: int) -> NDArray[np.float32]:
    """
    Filter the detections to only include the target class and those above a certain confidence level.
    Also converts the bounding box coordinates to the original frame size.
    """

    filtered = np.empty((detections.shape[2], 5), dtype=detections.dtype)
    i = 0

    for detection in detections[0, 0]:
        detection_confidence = detection[2]

        if detection_confidence < confidence:
            continue

        idx = int(detection[1])

        if idx != class_id:
            continue

        detection[3:] *= np.array([W, H, W, H])
        filtered[i, :-1] = detection[2:]
        filtered[i, -1] = detection_confidence

        i += 1

    return filtered[:i].copy()
