#!/bin/bash

# Base directory containing all subject folders
# BASE_DIR="/data-fast/james/triage/datasets/blue_orin_data/church-processed-train"
BASE_DIR="/data-fast/james/triage/equipleth/Camera_77GHzRadar_Plethysmography/cphys/rgb_files"

# Output directory for the extracted features
# OUTPUT_DIR="/data-fast/james/triage/datasets/blue_orin_data/church-processed-train/openface_features"
OUTPUT_DIR="/data-fast/james/triage/equipleth/Camera_77GHzRadar_Plethysmography/cphys/rgb_files/openface_features"
# mkdir -p "$OUTPUT_DIR"

# Iterate over all subdirectories in the base directory
for subject_dir in "$BASE_DIR"/*; do
    if [ -d "$subject_dir" ]; then  # Ensure it's a directory
        # Extract the subject number from the directory name
        subject_number=$(basename "$subject_dir")

        # Specify the output file name

        echo "Processing $subject_dir..."
        # Run feature extraction
        ../OpenFace/build/bin/FeatureExtraction -fdir "$subject_dir" -out_dir "$OUTPUT_DIR"
        # FeatureExtraction.exe -fdir "$subject_dir" -out_dir "$OUTPUT_DIR"
        # echo "Processed and saved as $output_file"
    fi
done

echo "Feature extraction completed."
