from typing import Dict, Optional, List

import numpy as np
from attr import dataclass
from core.types import (
    AnimationClipSettings,
)


@dataclass
class BoneData:
    name: str
    position: np.ndarray
    orientation: np.ndarray


@dataclass
class Keyframe:
    frame: int
    data: Dict[str, BoneData]


@dataclass
class AnimationClip:
    name: str
    framerate: int
    start_frame: int
    duration: int
    keyframes: List[Keyframe]


class AnimationEngine:
    def __init__(
        self,
        clips: Dict[str, AnimationClipSettings],
        max_episode_length: int,
    ):
        self.current_clip_name: Optional[str] = None

        self.clip_configs: Dict[str, AnimationClipSettings] = clips
        self.clips: Dict[str, AnimationClip] = {}

        self._max_episode_length: int = max_episode_length

        self._is_constructed: bool = False

    @property
    def current_clip(self) -> AnimationClip:
        assert (
            self._is_constructed
        ), "AnimationEngine not constructed: tried to access current_clip!"

        return self.clips[self.current_clip_name]

    def construct(self, current_clip: str) -> None:
        assert not self._is_constructed, "AnimationEngine already constructed!"
        assert (
            current_clip in self.clip_configs
        ), f"Clip {current_clip} not found in {self.clip_configs.keys()}"

        self.current_clip_name = current_clip

        for clip_name, clip_settings in self.clip_configs.items():
            keyframes: List[Keyframe] = []

            for keyframe_settings in clip_settings["keyframes"]:
                frame: int = keyframe_settings["frame"]
                data: Dict[str, BoneData] = {}

                for bone_data in keyframe_settings["data"]:
                    bone_name: str = bone_data["bone"]
                    position: np.ndarray = np.array(bone_data["position"])
                    orientation: np.ndarray = np.array(bone_data["orientation"])

                    data[bone_name] = BoneData(
                        name=bone_name,
                        position=position,
                        orientation=orientation,
                    )

                keyframe = Keyframe(
                    frame=frame,
                    data=data,
                )

                keyframes.append(keyframe)

            self.clips[clip_name] = AnimationClip(
                name=clip_name,
                framerate=clip_settings["framerate"],
                start_frame=clip_settings["beginning"],
                duration=clip_settings["duration"],
                keyframes=keyframes,
            )

        self._is_constructed = True

    # TODO: Implement clip reading functions (i.e. getting orientations/positions within a 0-1 range)
    #   Using the max episode length, we know the bounds of the requested progress. Therefore, we can figure out what
    #   keyframes to interpolate between and how to interpolate between them.
