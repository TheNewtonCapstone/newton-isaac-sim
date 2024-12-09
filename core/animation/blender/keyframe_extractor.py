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
        "FR_HFE",
        "FR_FKE",
        "FL_HAA",
        "FL_HFE",
        "FL_FKE",
        "HR_HAA",
        "HR_HFE",
        "HR_FKE",
        "HL_HAA",
        "HL_HFE",
        "HL_FKE",
    ]

    # List to store keyframe data
    keyframes_data = []

    # Get global animation start and end frames
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end

    # Access the armature object
    armature = bpy.data.objects.get(armature_name)

    framerate = bpy.context.scene.render.fps

    if armature and armature.type == "ARMATURE":
        print(f"Processing armature: {armature_name}")

        # Ensure armature is in pose mode
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="POSE")

        # Iterate through every frame in the global animation range
        for frame in range(start_frame, end_frame + 1):
            bpy.context.scene.frame_set(frame)  # Set the frame in the scene

            frame_data = []

            for bone_name in bone_names:
                bone = armature.pose.bones.get(bone_name)
                if bone:
                    world_matrix = armature.matrix_world @ bone.matrix

                    transform_keyframe = {
                        "bone": bone_name,
                        "position": list(world_matrix.to_translation()),
                        "orientation": list(world_matrix.to_quaternion()),
                    }

                    frame_data.append(transform_keyframe)

            keyframes_data.append(
                {
                    "frame": frame,
                    "data": frame_data,
                }
            )
    else:
        print(f"Armature {armature_name} not found, or not an armature.")

    return (
        framerate,
        keyframes_data,
        start_frame,
        end_frame - start_frame,
        len(bone_names),
    )


def keyframes_to_yaml(
    data: list,
    start_frame: int,
    duration: int,
    framerate: int,
    animation: str,
) -> str:
    return yaml.dump(
        {
            "name": animation,
            "start_frame": start_frame,
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
    framerate, rotations_dict, start_frame, duration, num_bones = (
        extract_keyframes_from_scene(armature_name)
    )

    print(
        f"Extracted rotation keyframes:\n  Bones: {num_bones}\n  Frames: {duration}\n  Total Keyframes: {duration * num_bones}\n  Framerate: {framerate}"
    )

    # Convert to YAML
    yaml_data = keyframes_to_yaml(
        rotations_dict,
        start_frame,
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
