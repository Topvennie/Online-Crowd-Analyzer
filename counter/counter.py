import concurrent.futures
import queue
import threading
from logging import getLogger
from typing import Callable

import cv2
import numpy as np
from cv2.typing import MatLike
from imutils.video import FPS
from numpy.typing import NDArray

from config import CounterConfig
from counter.trackableobject import TrackableObject


class Counter:

    def __init__(self, config: CounterConfig):
        self._logger = getLogger(__name__)

        self._config = config
        self._detector = config.detector.instance
        self._tracker = config.tracker.instance

        self._setup_input_output()

        self._fps: FPS
        self._running: bool

        self._detections_queue = queue.Queue()
        self._processed_frames = queue.Queue()

    def run(self, callback: Callable[[int], None]):
        """
        Run the counter.

        Args:
            callback (Callable[[int], None]): Callback function to call when a new object is counted.
        """
        self._running = True
        self._thread = threading.Thread(target=self._process, args=(callback,))
        self._thread.start()

    def get_frame(self) -> tuple[bool, MatLike]:
        """
        Get the latest processed frame.
        """
        if not self._running and self._processed_frames.empty():
            return False, np.zeros((self._H, self._W, 3), np.uint8)

        return True, self._processed_frames.get()

    def wait_until_finished(self):
        """
        Wait until the counter has finished.
        """
        self._thread.join()

    def _setup_input_output(self):
        # Input
        self._video = cv2.VideoCapture(self._config.input.file)
        if not self._video.isOpened():
            self._logger.error("Error: Cannot open video file.")
            exit(1)

        # Video properties
        self._W = int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._H = int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._video.get(cv2.CAP_PROP_FPS)

        # Output
        self._output: cv2.VideoWriter | None = None
        if self._config.output.file:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self._output = cv2.VideoWriter(self._config.output.file, fourcc, fps, (self._W, self._H), True)

    def _process(self, callback: Callable[[int], None]):
        self._running = True
        self._fps = FPS().start()

        thread = threading.Thread(target=self._track, args=(callback,))
        thread.start()

        ret, frame = self._video.read()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while ret:
                future = executor.submit(self._detect, frame)
                self._detections_queue.put(future)

                ret, frame = self._video.read()

        self._running = False

        thread.join()

    def _detect(self, frame: MatLike) -> tuple[MatLike, NDArray[np.float32]]:
        return frame, self._detector.detect_faces(frame)

    def _track(self, callback: Callable[[int], None]):
        trackable_objects: dict[int, TrackableObject] = {}
        total_counts = 0

        while True:
            future = self._detections_queue.get()
            (frame, detections) = future.result()

            objects = self._tracker.update(detections)
            trackable_objects = {k: v for k, v in trackable_objects.items() if k in objects}
            new_counts = 0

            for object_id, centroid in objects.items():
                trackable_object = trackable_objects.get(object_id, None)

                if not trackable_object:
                    trackable_object = TrackableObject(object_id, centroid)
                    trackable_objects[object_id] = trackable_object
                else:
                    trackable_object.centroid = centroid

                # Check if the object has been counted
                if not trackable_object.counted:
                    # Check if the object is in the counting area
                    if trackable_object.centroid[1] >= self._config.input.counting_line:
                        trackable_object.counted = True
                        new_counts += 1
                    # Check if the object didn't turn back after being counted
                    elif trackable_object.counted:
                        trackable_object.counted = False
                        new_counts -= 1

            if new_counts != 0:
                total_counts += new_counts
                callback(new_counts)

            if self._output or self._config.output.show:
                frame = self._draw(frame, trackable_objects, detections)
                if self._output:
                    self._output.write(frame)

                if self._config.output.show:
                    self._processed_frames.put(frame)

            self._fps.update()

            if not self._running and self._detections_queue.empty():
                break

        self._cleanup()
        self._log(total_counts)

    def _cleanup(self):
        """
        Cleans up the resources.
        """
        self._fps.stop()
        self._video.release()

        if self._output:
            self._output.release()

    def _log(self, total_counts: int):
        """
        Log some results.
        """
        self._logger.info("Counting finished.")
        self._logger.info(f"Total counts: {total_counts}")
        self._logger.info("Elapsed time: %.2f", self._fps.elapsed())
        self._logger.info("Approx. FPS: %.2f", self._fps.fps())

    def _draw(self, frame: MatLike, trackable_objects: dict[int, TrackableObject], detections: NDArray[np.float32]) -> MatLike:
        """
        Draw all objects on the frame.

        Blue rectangles are detections.
        Tracked objects have a green ID if they have been counted otherwise red.
        """
        # Draw all detections
        for detection in detections:
            (startX, startY, endX, endY) = detection[1:].astype(int)
            cv2.rectangle(
                frame,
                (startX, startY),
                (endX, endY),
                (255, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"Confidence: {detection[0]:.2f}",
                (startX, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
            )

        # Draw all tracked objects
        for trackable_object in trackable_objects.values():
            color = (0, 255, 0) if trackable_object.counted else (0, 0, 255)
            cv2.putText(
                frame,
                f"ID {trackable_object.object_id}",
                (trackable_object.centroid[0] - 20, trackable_object.centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            cv2.circle(
                frame,
                (trackable_object.centroid[0], trackable_object.centroid[1]),
                4, color, -1
            )

        # Draw the detection line
        cv2.putText(
            frame,
            "Detection Line",
            (10, self._config.input.counting_line - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )
        cv2.line(
            frame,
            (0, self._config.input.counting_line),
            (self._W, self._config.input.counting_line),
            (0, 0, 0), 3
        )

        # Draw the name
        cv2.putText(
            frame,
            self._config.name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4
        )

        return frame
