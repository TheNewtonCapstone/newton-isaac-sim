from typing import Dict, Optional, List

import numpy as np
from attr import dataclass
from core.types import (
    Settings,
)


@dataclass
class BoneData:
    name: str
    position: np.ndarray
    orientation: np.ndarray
    relative_angle: float


ArmatureData = Dict[str, BoneData]


@dataclass
class Keyframe:
    frame: float
    data: ArmatureData


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
        clips: Dict[str, Settings],
    ):
        self.current_clip_name: Optional[str] = None

        self.clip_configs: Dict[str, Settings] = clips
        self.clips: Dict[str, AnimationClip] = {}

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
                    relative_angle: float = bone_data["relative_angle"]

                    data[bone_name] = BoneData(
                        name=bone_name,
                        position=position,
                        orientation=orientation,
                        relative_angle=relative_angle,
                    )

                keyframe = Keyframe(
                    frame=frame,
                    data=data,
                )

                keyframes.append(keyframe)

            self.clips[clip_name] = AnimationClip(
                name=clip_name,
                framerate=clip_settings["framerate"],
                start_frame=clip_settings["start_frame"],
                duration=clip_settings["duration"],
                keyframes=keyframes,
            )

        self._is_constructed = True

    def get_current_clip_datas_ordered(
        self,
        progress: np.ndarray,
        joints_order: List[str],
        interpolate: bool = True,
    ) -> np.ndarray:
        """
        Get the armature data for the current clip at the given progress. Optionally interpolates between keyframes.
        Args:
            progress: The progress of the current episode, in the range [0, 1] for every vectorized agent.
            joints_order: List of joint names in the order they should be returned.
            interpolate: Whether to interpolate between keyframes (continuous result, assuming animation is continuous).

        Returns:
            A numpy array with shape (num_agents, num_bones, 8) containing the joint positions, orientations and relative angles for each agent.
        """
        clip_datas = self.get_clip_datas(self.current_clip_name, progress, interpolate)

        num_agents = len(clip_datas)
        num_bones = len(clip_datas[0])

        result = np.zeros((num_agents, num_bones, 8))

        for i, clip_data in enumerate(clip_datas):
            for j, bone_name in enumerate(joints_order):

                # TODO: Investigate why HR's bones are not proper
                #   HR is different from the other bones and gives weird rotation values (+- 90deg)

                # Quick fix for HR's bone structure being wonky
                if "HR" in bone_name:
                    bone_name = f"FR_{bone_name[3:]}"

                if bone_name not in clip_data:
                    continue

                bone_data = clip_data[bone_name]
                result[i, j] = np.concatenate(
                    [
                        bone_data.position,
                        bone_data.orientation,
                        [bone_data.relative_angle],
                    ]
                )

        return result

    def get_current_clip_datas(
        self, progress: np.ndarray, interpolate: bool = True
    ) -> List[ArmatureData]:
        """
        Get the armature data for the current clip at the given progress. Optionally interpolates between keyframes.
        Args:
            progress: The progress of the current episode, in the range [0, 1] for every vectorized agent.
            interpolate: Whether to interpolate between keyframes (continuous result, assuming animation is continuous).

        Returns:
            A list of armature data for each agent.
        """
        return self.get_clip_datas(self.current_clip_name, progress, interpolate)

    def get_clip_datas(
        self, clip_name: str, progress: np.ndarray, interpolate: bool = True
    ) -> List[ArmatureData]:
        """
        Get the armature data for the given clip at the given progress. Optionally interpolates between keyframes.
        Args:
            clip_name: The name of the clip to get data from.
            progress: The progress of the current episode, in the range [0, 1] for every vectorized agent.
            interpolate: Whether to interpolate between keyframes (continuous result, assuming animation is continuous).

        Returns:
            A list of armature data for each agent.
        """
        clip: AnimationClip = self.clips[clip_name]
        frames = progress * clip.duration

        data = []

        for frame in frames:
            data.append(self.get_clip_data(clip_name, frame, interpolate))

        return data

    def get_current_clip_data(
        self, frame: float, interpolate: bool = True
    ) -> ArmatureData:
        """
        Get the armature data for the current clip at the given frame. Optionally interpolates between keyframes.
        Args:
            frame: The frame to get data from (doesn't have to be within clip bounds). If it's not within bounds, this function will assume it's looping.
            interpolate: Whether to interpolate between keyframes (continuous result, assuming animation is continuous).

        Returns:
            Armature data for the current clip.
        """
        return self.get_clip_data(self.current_clip_name, frame, interpolate)

    def get_clip_data(
        self, clip_name: str, frame: float, interpolate: bool = True
    ) -> ArmatureData:
        """
        Get the armature data for the given clip at the given frame. Optionally interpolates between keyframes.
        Args:
            clip_name: The name of the clip to get data from.
            frame: The frame to get data from (doesn't have to be within clip bounds). If it's not within bounds, this function will assume it's looping.
            interpolate: Whether to interpolate between keyframes (continuous result, assuming animation is continuous).

        Returns:
            Armature data for the given clip.
        """
        clip: AnimationClip = self.clips[clip_name]
        keyframes = clip.keyframes

        if not interpolate:
            return keyframes[int(frame) % len(keyframes)].data

        this_keyframe = keyframes[int(frame) % len(keyframes)]
        next_keyframe = keyframes[int(frame + 1) % len(keyframes)]
        interpolated_data: ArmatureData = {}

        for bone_name, bone_data in this_keyframe.data.items():
            next_bone_data = next_keyframe.data[bone_name]

            from core.utils.math import lerp, quat_slerp_n

            position = lerp(bone_data.position, next_bone_data.position, frame % 1)
            orientation = quat_slerp_n(
                bone_data.orientation, next_bone_data.orientation, frame % 1
            )
            relative_angle = lerp(
                bone_data.relative_angle, next_bone_data.relative_angle, frame % 1
            )

            interpolated_data[bone_name] = BoneData(
                name=bone_name,
                position=position,
                orientation=orientation,
                relative_angle=relative_angle,
            )

        return interpolated_data
