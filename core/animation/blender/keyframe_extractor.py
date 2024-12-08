import bpy
import argparse
import sys
import yaml


def extract_keyframes_from_scene(
    armature_name: str = "Armature",
    euler: bool = False,
) -> tuple:
    # List of bone names you are interested in
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
                    rotation_quaternion = world_matrix.to_quaternion()

                    rotation_keyframe = {
                        "bone": bone_name,
                        "quaternion": list(rotation_quaternion[:]),
                    }

                    if euler:
                        rotation_euler = world_matrix.to_euler()

                        # Convert Euler angles to degrees
                        rotation_euler[:] = [e * 180 / 3.14159 for e in rotation_euler]

                        rotation_keyframe["euler"] = list(rotation_euler[:])

                    frame_data.append(rotation_keyframe)

            keyframes_data.append(
                {
                    "frame": frame,
                    "data": frame_data,
                }
            )
    else:
        print(f"Armature {armature_name} not found, or not an armature.")

    return keyframes_data, end_frame - start_frame, len(bone_names)


def list_to_yaml(data: list) -> str:
    return yaml.dump(
        {
            "keyframes": data,
        }
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
        help="Name of the Armature to extract keyframes from.",
    )
    parser.add_argument(
        "--keyframes",
        required=False,
        help="Path to the file to save the keyframes to.",
    )

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    return args


def main():
    args = setup_args()
    armature_name = args.armature
    keyframes_path = args.keyframes

    # Extract keyframes
    rotations_dict, length, num_bones = extract_keyframes_from_scene(armature_name)

    print(
        f"Extracted rotation keyframes:\n  Bones: {num_bones}\n  Frames: {length}\n  Total Keyframes: {length * num_bones}"
    )

    # Convert to YAML
    yaml_data = list_to_yaml(rotations_dict)

    # Save to file
    with open(keyframes_path, "w") as file:
        file.write(yaml_data)

    print(f"Keyframes saved to: {keyframes_path}")


# Ensure the script only runs when executed directly
if __name__ == "__main__":
    main()
