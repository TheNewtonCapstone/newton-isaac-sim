import argparse


def get_blender_version_from_blend(filepath):
    """
    Reads the Blender version from the header of a .blend file.

    Args:
        filepath (str): Path to the .blend file.

    Returns:
        tuple: (major_version, minor_version, patch_version) or None if not a valid .blend file.
    """
    try:
        with open(filepath, "rb") as f:
            # Read the first 12 bytes of the header
            header = f.read(12)

            # Validate the file type (starts with 'BLENDER')
            if not header.startswith(b"BLENDER"):
                raise ValueError("Not a valid .blend file.")

            # Blender version is stored in bytes 9-11
            major_version = header[9] - ord(b"0")
            minor_version = header[10] - ord(b"0")
            patch_version = header[11] - ord(b"0")

            return major_version, minor_version, patch_version

    except Exception as e:
        print(f"Error reading .blend file: {e}")
        return -1, -1, -1


def semantic_version_to_tuple(version: str) -> tuple:
    """
    Converts a semantic version string to a tuple of integers.

    Args:
        version (str): Semantic version string (e.g., '2.83.0').

    Returns:
        tuple: (major, minor, patch) version numbers.
    """
    return tuple(map(int, version.split(".")))


def are_versions_compatible(blender_version: tuple, file_version: tuple):
    """
    Compares two Blender versions to check compatibility.

    Args:
        blender_version (tuple): Blender version to check against.
        file_version (tuple): Blender version from the file.

    Returns:
        bool: True if the file version is compatible with the Blender version.
    """
    bv_major, bv_minor, bv_patch = blender_version
    fv_major, fv_minor, fv_patch = file_version

    # If any are -1, return False
    if -1 in blender_version or -1 in file_version:
        return False

    # Check major version compatibility
    if bv_major != fv_major:
        return False

    # Check minor version compatibility
    if bv_minor < fv_minor:
        return False

    # Check patch version compatibility
    if bv_minor == fv_minor and bv_patch < fv_patch:
        return False

    return True


def setup_args() -> argparse.Namespace:
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract Blender version from a .blend file."
    )
    parser.add_argument(
        "file",
        help="Path to the .blend file",
    )
    parser.add_argument(
        "--blender-version",
        required=False,
        help="Blender version to check against.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print additional information.",
    )

    args = parser.parse_args()
    return args


def main():
    args = setup_args()
    file_path = args.file
    verbose = args.verbose
    check_blender_version = args.blender_version is not None

    # Get the Blender version from the provided file
    fv_major, fv_minor, fv_patch = get_blender_version_from_blend(file_path)

    # Check compatibility with the provided Blender version
    if check_blender_version:
        bv_major, bv_minor, bv_patch = semantic_version_to_tuple(args.blender_version)
        compatible = are_versions_compatible(
            (bv_major, bv_minor, bv_patch),
            (fv_major, fv_minor, fv_patch),
        )

        if verbose:
            print(f"File version: {fv_major}.{fv_minor}.{fv_patch}")
            print(f"Blender version: {bv_major}.{bv_minor}.{bv_patch}")
            print(f"Compatible: {compatible}")
        else:
            print("Compatible" if compatible else "Incompatible", end=" ")
            print(f"{fv_major}.{fv_minor}.{fv_patch}")
    else:
        if verbose:
            print(f"File version: {fv_major}.{fv_minor}.{fv_patch}")
        else:
            print(f"{fv_major}.{fv_minor}.{fv_patch}")


if __name__ == "__main__":
    main()
