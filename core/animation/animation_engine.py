from typing import Dict, Optional, List

import torch
from ..types import Config, EpisodeLength
from .types import AnimationClip, Keyframe, BoneData, ArmatureData
from ..base import BaseObject
from ..logger import Logger
from ..universe import Universe


class AnimationEngine(BaseObject):
    def __init__(
        self,
        universe: Universe,
        current_clip_config: Config,
    ):
        super().__init__(universe=universe)

        self.clip_config: Config = current_clip_config
        self.clip: Optional[AnimationClip] = None

    def construct(self) -> None:
        super().construct()

        frame_dt = 1 / self.clip_config["framerate"]

        Logger.info(
            f"Constructing AnimationEngine with frame_dt: {frame_dt} and current_clip_name: {self.clip_config['name']}"
        )

        saved_keyframes: List[Config] = self.clip_config["keyframes"]
        keyframes: List[Keyframe] = []

        for keyframe_settings in saved_keyframes:
            frame: int = keyframe_settings["frame"]
            saved_data: List[Config] = keyframe_settings["data"]

            data: Dict[str, BoneData] = {}

            for i, bone_data in enumerate(saved_data):
                bone_name: str = bone_data["bone"]
                position: torch.Tensor = torch.tensor(bone_data["position"])
                orientation: torch.Tensor = torch.tensor(bone_data["orientation"])

                relative_angle: float = bone_data["relative_angle"]
                previous_relative_angle: float = saved_keyframes[frame - 1]["data"][i][
                    "relative_angle"
                ]
                relative_angle_velocity: float = (
                    relative_angle - previous_relative_angle
                ) / frame_dt

                data[bone_name] = BoneData(
                    name=bone_name,
                    position=position,
                    orientation=orientation,
                    relative_angle=relative_angle,
                    relative_angle_velocity=relative_angle_velocity,
                )

            keyframe = Keyframe(
                frame=frame,
                data=data,
            )

            keyframes.append(keyframe)

        duration_in_seconds = self.clip_config["duration"] * frame_dt

        self.clip = AnimationClip(
            name=self.clip_config["name"],
            framerate=self.clip_config["framerate"],
            duration=self.clip_config["duration"],
            duration_in_seconds=duration_in_seconds,
            keyframes=keyframes,
        )

        self._is_constructed = True

    def post_construct(self):
        super().post_construct()

        self._is_post_constructed = True

    def get_multiple_clip_data_at_seconds(
        self,
        seconds: EpisodeLength,
        joints_order: List[str],
        interpolate: bool = True,
    ) -> torch.Tensor:
        """
        Get the armature data for the current clip at the given progress. Optionally interpolates between keyframes.
        Args:
            seconds: The progress of the current episode, in seconds, for every vectorized agent.
            joints_order: List of joint names in the order they should be returned.
            interpolate: Whether to interpolate between keyframes (continuous result, assuming animation is continuous).

        Returns:
            A tensor with shape (num_agents, num_bones, 9) containing the joint positions, orientations, relative angles and relative angle velocities for each agent.
        """
        assert (
            self.is_fully_constructed
        ), "AnimationEngine not constructed: tried to get multiple clip data!"

        clip_datas = self.get_clip_data_at_seconds(
            seconds,
            interpolate,
        )

        num_agents = len(clip_datas)
        num_bones = len(clip_datas[0])

        result = torch.zeros((num_agents, num_bones, 9))

        for i, clip_data in enumerate(clip_datas):
            for j, bone_name in enumerate(joints_order):
                if bone_name not in clip_data:
                    continue

                bone_data = clip_data[bone_name]
                result[i, j, :] = torch.cat(
                    [
                        bone_data.position,
                        bone_data.orientation,
                        torch.tensor([bone_data.relative_angle]),
                        torch.tensor([bone_data.relative_angle_velocity]),
                    ],
                )

        return result

    def get_clip_data_at_seconds(
        self,
        second: EpisodeLength,
        interpolate: bool = True,
    ) -> List[ArmatureData]:
        """
        Get the armature data for the current clip at the given progress. Optionally interpolates between keyframes.
        Args:
            second: The progress of the current episode, in seconds, for every vectorized agent.
            interpolate: Whether to interpolate between keyframes (continuous result, assuming animation is continuous).

        Returns:
            A list of armature data for each agent.
        """
        assert (
            self.is_fully_constructed
        ), "AnimationEngine not constructed: tried to get clip data at seconds!"

        frames = second.cpu() * self.clip.framerate

        data = []

        for frame in frames:
            data.append(self.get_clip_data_at_frame(frame, interpolate))

        return data

    def get_clip_data_at_frame(
        self,
        frame: float,
        interpolate: bool = True,
    ) -> ArmatureData:
        """
        Get the armature data for the current clip at the given frame. Optionally interpolates between keyframes.
        Args:
            frame: The frame to get data from (doesn't have to be within clip bounds). If it's not within bounds, this function will assume it's looping.
            interpolate: Whether to interpolate between keyframes (continuous result, assuming animation is continuous).

        Returns:
            Armature data for the given clip.
        """
        assert (
            self.is_fully_constructed
        ), "AnimationEngine not constructed: tried to get clip data at frame!"

        keyframes = self.clip.keyframes

        if not interpolate:
            return keyframes[int(frame) % len(keyframes)].data

        this_keyframe = keyframes[int(frame) % len(keyframes)]
        next_keyframe = keyframes[int(frame + 1) % len(keyframes)]
        interpolated_data: ArmatureData = {}

        for bone_name, bone_data in this_keyframe.data.items():
            next_bone_data = next_keyframe.data[bone_name]

            from core.utils.math import lerp, quat_slerp_t

            position = lerp(
                bone_data.position,
                next_bone_data.position,
                frame % 1,
            )
            orientation = quat_slerp_t(
                bone_data.orientation,
                next_bone_data.orientation,
                frame % 1,
            )
            relative_angle = lerp(
                bone_data.relative_angle,
                next_bone_data.relative_angle,
                frame % 1,
            )
            relative_angle_velocity = lerp(
                bone_data.relative_angle_velocity,
                next_bone_data.relative_angle_velocity,
                frame % 1,
            )

            interpolated_data[bone_name] = BoneData(
                name=bone_name,
                position=position,
                orientation=orientation,
                relative_angle=relative_angle,
                relative_angle_velocity=relative_angle_velocity,
            )

        return interpolated_data
