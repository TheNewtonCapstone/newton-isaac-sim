#!/bin/bash

# Purpose
# Wraps python script to extract keyframes from a sequence of images

# .blend file from arguments, make sure the argument is passed and the file exists
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <blend_file> <output_dir>"
    exit 1
fi

# Check if at root directory of the project
ROOT_PRJ_DIRECTORY=$(git rev-parse --show-toplevel)

if [ ! -f "scripts/keyframe_extractor.sh" ]; then
    echo "Please run this script from the root directory of the project (${ROOT_PRJ_DIRECTORY})."
    exit 1
fi


BLEND_FILE=$1
OUTPUT_DIR=$2
if [ ! -f "$BLEND_FILE" ]; then
    echo "Blender file does not exist: $BLEND_FILE"
    exit 1
fi

# Finds blender executable
BLENDER_EXECUTABLE=$(which blender)
if [ -z "$BLENDER_EXECUTABLE" ]; then
    echo "Blender executable not found. Please install Blender."
    exit 1
fi

# Get blender version, reading until the first newline
BLENDER_VERSION=$($BLENDER_EXECUTABLE --version | head -n 1 | cut -d " " -f 2)
echo "Found Blender version: $BLENDER_VERSION"

# Check if the version checker python script exists
VERSION_CHECKER_SCRIPT="core/animation/blender/version_checker.py"
if [ ! -f "$VERSION_CHECKER_SCRIPT" ]; then
    echo "Version checker script not found: $VERSION_CHECKER_SCRIPT. Are you in the root directory of the project?"
    exit 1
fi
echo "Found version checker script."

# Check if the file version is compatible with the blender's version (version_checker outputs <Compatible|Incompatible> <version>)
VERSION_CHECKER_OUT=$(python $VERSION_CHECKER_SCRIPT "$BLEND_FILE" --blender-version "$BLENDER_VERSION")
VERSION_CHECK=$(echo "$VERSION_CHECKER_OUT" | cut -d " " -f 1)
FILE_VERSION=$(echo "$VERSION_CHECKER_OUT" | cut -d " " -f 2)

if [ "$VERSION_CHECK" != "Compatible" ]; then
    echo "File (version $FILE_VERSION) is incompatible with found Blender. Please use a compatible version."
    exit 1
fi
echo "File (version $FILE_VERSION) is compatible with found Blender."

# Extract keyframes using extractor python script
KEYFRAME_EXTRACTOR_SCRIPT="core/animation/blender/keyframe_extractor.py"
if [ ! -f "$KEYFRAME_EXTRACTOR_SCRIPT" ]; then
    echo "Keyframe extractor script not found: $KEYFRAME_EXTRACTOR_SCRIPT. Are you in the root directory of the project?"
    exit 1
fi
echo "Found keyframe extractor script."

echo "Extracting keyframes from $BLEND_FILE..."

# Get only the file name from the path (without the path and extension)
BLEND_FILE_NAME=$(basename "$BLEND_FILE" | cut -d "." -f 1)

OUTPUT_FILE="$OUTPUT_DIR/$BLEND_FILE_NAME.keyframes.yaml"
exec $BLENDER_EXECUTABLE --python-use-system-env --background --quiet "$BLEND_FILE" --python $KEYFRAME_EXTRACTOR_SCRIPT \
  -- --armature "Armature" --output "$OUTPUT_FILE" --animation "$BLEND_FILE_NAME"
