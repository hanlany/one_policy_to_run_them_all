from dataclasses import dataclass
from pathlib import Path
from typing import Type

import gymnasium as gym


@dataclass(frozen=True)
class RobotSpec:
    cls: Type
    long_name: str
    short_name: str

    @property
    def module_name(self) -> str:
        return self.cls.__module__.rsplit(".", 1)[0]

    @property
    def data_dir(self) -> Path:
        module = __import__(self.module_name, fromlist=["__file__"])
        return Path(module.__file__).resolve().parent / "data"

    def validate(self):
        if not self.long_name:
            raise ValueError(f"Robot spec for {self.cls.__name__} is missing a long_name.")
        if not self.short_name:
            raise ValueError(f"Robot spec for {self.cls.__name__} is missing a short_name.")
        if getattr(self.cls, "LONG_NAME", None) != self.long_name:
            raise ValueError(f"Robot spec long_name does not match {self.cls.__name__}.LONG_NAME.")
        if getattr(self.cls, "SHORT_NAME", None) != self.short_name:
            raise ValueError(f"Robot spec short_name does not match {self.cls.__name__}.SHORT_NAME.")
        return self


class BaseMujocoRobotEnv(gym.Env):
    ROBOT_SPEC: RobotSpec | None = None

    @classmethod
    def robot_spec(cls) -> RobotSpec:
        if cls.ROBOT_SPEC is None:
            return RobotSpec(cls=cls, long_name=cls.LONG_NAME, short_name=cls.SHORT_NAME).validate()
        return cls.ROBOT_SPEC.validate()
