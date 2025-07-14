#!/bin/bash
# Download index files from OpenAI Video-Pre-Training dataset
# https://github.com/openai/Video-Pre-Training
#
# This script downloads specific JSON index files containing metadata
# for the OpenAI Video-Pre-Training dataset from Microsoft Azure blob storage.
#
# Usage:
#   ./download_index_files.sh [output_dir]
#
# Arguments:
#   output_dir: Directory to save downloaded files (default: data/open_ai_index_files)
#
# Example:
#   ./download_index_files.sh /path/to/custom/directory

# Set output directory, use default if not provided
output_dir="${1:-data/open_ai_index_files}" 
mkdir -p $output_dir

# List of index files to download
# These files contain metadata for different video ranges (6xx, 7xx, etc.)
index_files=(
    "all_6xx_Jun_29.json"    
    "all_7xx_Apr_6.json"     
    "all_8xx_Jun_29.json"    
    "all_9xx_Jun_29.json"    
    "all_10xx_Jun_29.json"   
)

# Download each index file from Azure blob storage
for index_file in "${index_files[@]}"; do
    echo "Downloading $index_file..."
    wget https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/$index_file -O $output_dir/$index_file
done

echo "Download complete. Files saved to: $output_dir"
