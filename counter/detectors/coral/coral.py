from typing import Sequence

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from cv2.typing import MatLike
from numpy.typing import NDArray

from counter.detector import Detector

MODEL_PATH = "counter/detectors/coral/"

CLASSES_INDEX = 0


class Coral(Detector):

    def __init__(
        self,
        model: str = "yolov8n",
        confidence: float = 0.6,
        iou: float = 0.5,
        driver_path: str = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"
    ) -> None:
        super().__init__()

        self._confidence = confidence
        self._iou = iou

        self._interpreter = self._make_interpreter(MODEL_PATH + model + ".tflite", driver_path)
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        self._height: int = self._input_details[0]["shape"][1]
        self._width: int = self._input_details[0]["shape"][2]

        self._letterbox = LetterBox((self._width, self._height))

    def detect_faces(self, frame: MatLike) -> NDArray[np.float32]:
        # Preprocess
        img_data = self._preprocess(frame)

        # Run inference
        self._interpreter.set_tensor(self._input_details[0]["index"], img_data)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])

        # Postprocess
        scale, zero_point = self._output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale
        output[:, [0, 2]] *= self._width
        output[:, [1, 3]] *= self._height

        return self._postprocess(output)

    def _preprocess(self, frame: MatLike) -> MatLike:
        # To scale the output for drawing bounding boxes
        self.img_height, self.img_width = frame.shape[:2]

        image = self._letterbox(frame)

        image = [image]
        image = np.stack(image)
        image = image[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(image)
        image = img.astype(np.float32)
        image = image / 255
        img_data = image.transpose((0, 2, 3, 1))
        scale, zero_point = self._input_details[0]["quantization"]
        img_data_int8 = (img_data / scale + zero_point).astype(np.int8)

        return img_data_int8

    def _postprocess(self, output: NDArray) -> NDArray[np.float32]:
        boxes: Sequence[list[np.float64]] = []
        confidences: Sequence[np.float32] = []
        class_ids: list[np.int64] = []
        for pred in output:
            pred = np.transpose(pred)
            for box in pred:
                idx = np.argmax(box[4:])
                confidence = box[idx + 4]
                class_id = idx

                if class_id == CLASSES_INDEX and confidence >= self._confidence:
                    x, y, w, h = box[:4]
                    x1 = x - w / 2
                    y1 = y - h / 2
                    boxes.append([x1, y1, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Type error can be solved by replacing np.float 64 and np.float32 with float
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self._confidence, self._iou)  # type: ignore

        result = np.empty((len(indices), 5), dtype=np.float32)
        for index, i in enumerate(indices):
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            gain = min(self._width / self.img_width, self._height / self.img_height)
            pad = (
                round((self._width - self.img_width * gain) / 2 - 0.1),
                round((self._height - self.img_height * gain) / 2 - 0.1),
            )
            box[0] = (box[0] - pad[0]) / gain
            box[1] = (box[1] - pad[1]) / gain
            box[2] = box[2] / gain
            box[3] = box[3] / gain
            confidence = confidences[i]
            class_id = class_ids[i]

            result[index, :-1] = box
            result[index, -1] = confidence

        return result

    def _make_interpreter(self, model_path: str, driver_path: str):
        delegates = tflite.load_delegate(driver_path, {})
        return tflite.Interpreter(model_path=model_path, experimental_delegates=[delegates])


class LetterBox:
    """Resizes and reshapes images while maintaining aspect ratio by adding padding, suitable for YOLO models."""

    def __init__(self, new_shape: tuple[int, int]):
        """Initializes LetterBox with parameters for reshaping and transforming image while maintaining aspect ratio."""
        self._new_shape = new_shape

    def __call__(self, image: MatLike):
        """Return updated image with added border."""

        img = image
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self._new_shape[0] / shape[0], self._new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self._new_shape[1] - new_unpad[0], self._new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides to keep it in the center
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        return img
