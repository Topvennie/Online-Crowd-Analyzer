from __future__ import annotations

from enum import StrEnum
from logging import getLogger
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from counter.detector import Detector
from counter.detectors.hog.hog import HOG
from counter.detectors.mobile.mobile import MobileNetSSD
from counter.detectors.yolo.yolo import Yolo
from counter.tracker import Tracker
from counter.trackers.centroidtracker.centroidtracker import CentroidTracker

logger = getLogger(__name__)


class InputConfig(BaseModel):
    """
    Configuration for the input video.
    """

    file: str  # Path to the video file
    counting_line: int  # Y coordinate of the counting line. Only people crossing this line downwards are counted


class OutputConfig(BaseModel):
    """
    Configuration for the output video.
    """

    file: str | None  # Path to the output video file. None if no output video is needed
    show: bool  # Show the output video live


class DetectorModelConfig(StrEnum):
    """
    All available detector models.
    """

    MOBILENETSSD = "mobileNetSSD"
    HOG = "hog"
    YOLO = "yolo"

    def create_instance(self, arguments: dict[str, Any]) -> Detector:
        models = {
            DetectorModelConfig.MOBILENETSSD: MobileNetSSD,
            DetectorModelConfig.HOG: HOG,
            DetectorModelConfig.YOLO: Yolo
        }

        return models[self](**arguments)


class DetectorConfig(BaseModel):
    """
    Configuration for the detector.

    Supports any number of arguments that are passed to the detector model.
    """

    model: DetectorModelConfig  # The model to use to detect people
    arguments: dict[str, Any] = {}  # Arguments to pass to the model

    @property
    def instance(self) -> Detector:
        return self.model.create_instance(self.arguments)


class TrackerModelConfig(StrEnum):
    """
    All available tracker models.
    """

    CENTROID = "centroid"
    # SORT = "sort"
    # DEEPSORT = "deepsort"

    def create_instance(self, arguments: dict[str, Any]) -> Tracker:
        models = {
            TrackerModelConfig.CENTROID: CentroidTracker,
            # TrackerModelConfig.SORT: SortTracker,
            # TrackerModelConfig.DEEPSORT: DeepSortTracker,
        }

        return models[self](**arguments)


class TrackerConfig(BaseModel):
    """
    Configuration for the detector.

    Supports any number of arguments that are passed to the tracker model.
    """

    model: TrackerModelConfig  # The model to use to track people
    arguments: dict[str, Any] = {}  # Arguments to pass to the model

    @property
    def instance(self):
        return self.model.create_instance(self.arguments)


class CounterConfig(BaseModel):
    """
    Configuration used to create a single counter.
    """

    name: str  # Name of the counter
    input: InputConfig  # Input configuration
    output: OutputConfig  # Output configuration
    detector: DetectorConfig  # Detector configuration
    tracker: TrackerConfig  # Tracker configuration


class ShowConfig(BaseModel):
    """
    Configuration for the counters that should show the output.
    """

    wait: int  # Time to wait for a key press before moving to the next frame


class Config(BaseModel):
    """
    Main Configuration class.
    """

    add: list[CounterConfig] = []  # Counters that watch for people to add
    remove: list[CounterConfig] = []  # Counters that watch for people to remove
    show: ShowConfig  # Configuration for the counters that should show the output

    @classmethod
    def load_from_yaml(cls, file_path: str) -> Config:
        with open(file_path, "r") as f:
            settings_data = yaml.safe_load(f)

        try:
            cls.model_validate(settings_data)
        except ValidationError as e:
            logger.error(f"Error: Invalid configuration file: {file_path}")
            logger.error(f"Error: {e}")
            exit(1)

        return cls(**settings_data)
