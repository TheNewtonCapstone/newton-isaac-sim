import math

import bpy
import argparse
import sys
import yaml


def extract_keyframes_from_scene(
    armature_name: str = "Armature",
) -> tuple:
    # List of bone names we are interested in
    bone_names = [
        "FR_HAA",
        "FL_HAA",
        "HR_HAA",
        "HL_HAA",
        "FR_HFE",
        "HR_HFE",
        "FL_HFE",
        "HL_HFE",
        "FR_KFE",
        "FL_KFE",
        "HR_KFE",
        "HL_KFE",
    ]
    num_bones = len(bone_names)

    # Get global animation start and end frames
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end

    # Access the armature object
    armature = bpy.data.objects.get(armature_name)
    if not armature:
        print(f"Armature {armature_name} not found!")
        return (0, [], 0, 0, 0)
    elif armature and not armature.type == "ARMATURE":
        print(f"Object {armature_name} is not an armature!")
        return (0, [], 0, 0, 0)

    print(f"Processing armature: {armature_name}")

    framerate = bpy.context.scene.render.fps

    # Ensure armature is in pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    # List to store non-transformed keyframe data
    raw_animation = []

    # Iterate through every frame in the global animation range
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)  # Set the frame in the scene

        raw_keyframe_data = []

        for bone_name in bone_names:
            bone = armature.pose.bones.get(bone_name)
            if bone:
                parent = bone.parent

                world_matrix = armature.matrix_world @ bone.matrix

                # Compute relative orientation
                if parent:
                    parent_world_matrix = armature.matrix_world @ parent.matrix
                    relative_matrix = parent_world_matrix.inverted() @ world_matrix
                    relative_quaternion = relative_matrix.to_quaternion()
                else:
                    # No parent, so the matrix is already in world space
                    relative_quaternion = world_matrix.to_quaternion()

                relative_euler = relative_quaternion.to_euler()
                relative_euler[:] = [math.degrees(e) for e in relative_euler]

                raw_transform_keyframe = {
                    "bone": bone_name,
                    "position": list(world_matrix.to_translation()),
                    "orientation": list(world_matrix.to_quaternion()),
                    "relative_angles": list(relative_euler),
                }

                raw_keyframe_data.append(raw_transform_keyframe)

        raw_animation.append(
            {
                "frame": frame,
                "data": raw_keyframe_data,
            }
        )

    # List to store transformed keyframe data
    animation = []

    angle_ranges = [[[0] * 2] * 3] * num_bones

    # Iterate through every frame in the global animation range and finds the relative angles that changed the most,
    # that's the axis of rotation for the bone
    for keyframe in raw_animation:
        frame = keyframe["frame"]
        raw_keyframe_data = keyframe["data"]

        bpy.context.scene.frame_set(frame)

        for i, bone_data in enumerate(raw_keyframe_data):
            for j, angle in enumerate(bone_data["relative_angles"]):
                angle_ranges[i][j][0] = min(angle_ranges[i][j][0], angle)
                angle_ranges[i][j][1] = max(angle_ranges[i][j][1], angle)

    # Builds the final animation data using the axis of rotation for each bone (chosen by the relative angle that
    # changed the most)
    for keyframe in raw_animation:
        frame = keyframe["frame"]
        raw_keyframe_data = keyframe["data"]

        bpy.context.scene.frame_set(frame)

        keyframe_data = []

        for i, bone_data in enumerate(raw_keyframe_data):
            transform_keyframe = {
                "bone": bone_data["bone"],
                "position": bone_data["position"],
                "orientation": bone_data["orientation"],
            }

            chosen_relative_angle = 0
            largest_angle_range_size = -360
            for j, angle in enumerate(bone_data["relative_angles"]):
                angle_range = angle_ranges[i][j]
                angle_range_size = angle_range[1] - angle_range[0]

                if angle_range_size > largest_angle_range_size:
                    largest_angle_range_size = angle_range_size
                    chosen_relative_angle = angle

            adjustment = 0

            if "R_HAA" in bone_data["bone"]:
                adjustment = -90
            elif "L_HAA" in bone_data["bone"]:
                adjustment = 90
            elif "_HFE" in bone_data["bone"]:
                adjustment = 90

            transform_keyframe["relative_angle"] = chosen_relative_angle + adjustment

            keyframe_data.append(transform_keyframe)

        animation.append(
            {
                "frame": frame,
                "data": keyframe_data,
            }
        )

    return (
        framerate,
        animation,
        end_frame - start_frame,
        len(bone_names),
    )


def keyframes_to_yaml(
    data: list,
    duration: int,
    framerate: int,
    name: str,
) -> str:
    return yaml.dump(
        {
            "name": name,
            "duration": duration,
            "framerate": framerate,
            "keyframes": data,
        },
        sort_keys=False,
    )


def setup_args() -> argparse.Namespace:
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract keyframes from a Blender scene."
    )
    parser.add_argument(
        "--armature",
        required=False,
        default="Armature",
        type=str,
        help="Name of the Armature to extract keyframes from.",
    )
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        help="Path to the file to save the keyframes to.",
    )
    parser.add_argument(
        "--animation",
        required=True,
        type=str,
        help="Name of the animation (to be used in the output YAML).",
    )

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    return args


def main():
    args = setup_args()
    armature_name = args.armature
    output_path = args.output
    animation_name = args.animation

    # Extract keyframes
    framerate, rotations_dict, duration, num_bones = extract_keyframes_from_scene(
        armature_name
    )

    print(
        f"Extracted rotation keyframes:\n  Bones: {num_bones}\n  Frames: {duration}\n  Total Keyframes: {duration * num_bones}\n  Framerate: {framerate}"
    )

    # Convert to YAML
    yaml_data = keyframes_to_yaml(
        rotations_dict,
        duration,
        framerate,
        animation_name,
    )

    # Save to file
    with open(output_path, "w") as file:
        file.write(yaml_data)

    print(f"Keyframes saved to: {output_path}")


# Ensure the script only runs when executed directly
if __name__ == "__main__":
    main()
