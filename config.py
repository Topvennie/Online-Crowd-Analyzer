from __future__ import annotations

from logging import getLogger

import yaml
from pydantic import BaseModel, ValidationError

logger = getLogger(__name__)


class Output(BaseModel):
    file: str | None
    show: bool
    realtime: bool


class Config(BaseModel):
    video: str
    output: Output

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
