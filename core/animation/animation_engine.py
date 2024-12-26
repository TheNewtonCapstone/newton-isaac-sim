from typing import Dict, Optional, List

import torch
from core.types import (
    Settings,
    Progress,
)
from .types import AnimationClip, Keyframe, BoneData, ArmatureData


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

        frame_dt = 1 / self.clip_configs[current_clip]["framerate"]

        for clip_name, clip_settings in self.clip_configs.items():
            saved_keyframes: List[Settings] = clip_settings["keyframes"]
            keyframes: List[Keyframe] = []

            for keyframe_settings in saved_keyframes:
                frame: int = keyframe_settings["frame"]
                saved_data: List[Settings] = keyframe_settings["data"]

                data: Dict[str, BoneData] = {}

                for i, bone_data in enumerate(saved_data):
                    bone_name: str = bone_data["bone"]
                    position: torch.Tensor = torch.tensor(bone_data["position"])
                    orientation: torch.Tensor = torch.tensor(bone_data["orientation"])

                    relative_angle: float = bone_data["relative_angle"]
                    previous_relative_angle: float = saved_keyframes[frame - 1]["data"][i]["relative_angle"]
                    relative_angle_velocity: float = (relative_angle - previous_relative_angle) / frame_dt

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

            duration_in_seconds = clip_settings["duration"] * frame_dt

            self.clips[clip_name] = AnimationClip(
                name=clip_name,
                framerate=clip_settings["framerate"],
                duration=clip_settings["duration"],
                duration_in_seconds=duration_in_seconds,
                keyframes=keyframes,
            )

        self._is_constructed = True

    def get_multiple_clip_data_at_seconds(
        self,
        seconds: Progress,
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
        clip_datas = self.get_clip_data_at_seconds(
            self.current_clip_name, seconds, interpolate,
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
        clip_name: str,
        second: Progress,
        interpolate: bool = True,
    ) -> List[ArmatureData]:
        """
        Get the armature data for the given clip at the given progress. Optionally interpolates between keyframes.
        Args:
            clip_name: The name of the clip to get data from.
            second: The progress of the current episode, in seconds, for every vectorized agent.
            interpolate: Whether to interpolate between keyframes (continuous result, assuming animation is continuous).

        Returns:
            A list of armature data for each agent.
        """
        clip: AnimationClip = self.clips[clip_name]
        frames = second.cpu() * clip.framerate

        data = []

        for frame in frames:
            data.append(self.get_clip_data_at_frame(clip_name, frame, interpolate))

        return data

    def get_clip_data_at_frame(
        self,
        clip_name: str,
        frame: float,
        interpolate: bool = True,
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
