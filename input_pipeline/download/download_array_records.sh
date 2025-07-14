#!/bin/bash

# Download and extract array records from Hugging Face
# 
# This script performs a two-step process:
# 1. Downloads compressed array records from a Hugging Face dataset repository
# 2. Extracts the compressed tar files in parallel for better performance
#
# Prerequisites:
# - huggingface-cli must be installed and configured
# - Sufficient disk space for both compressed and uncompressed data
#
# Usage:
#   ./download_array_records.sh [hf_download_dir] [final_dataset_dir]
#
# Arguments:
#   hf_download_dir    - Directory to store compressed downloads (default: data/minecraft_arrayrecords_compressed)
#   final_dataset_dir  - Directory for extracted array records (default: data/minecraft_arrayrecords)

# Set default directories if not provided as arguments
hf_download_dir="${1:-data/minecraft_arrayrecords_compressed}" 
final_dataset_dir="${2:-data/minecraft_arrayrecords}"          

mkdir -p $hf_download_dir
mkdir -p $final_dataset_dir

# Step 1: Download compressed dataset from Hugging Face
echo "Starting download from Hugging Face..."
repo_id=avocadoali/open_ai_minecraft_arrayrecords_chunked
start_time_hf_download=$(date +%s)

HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_SYMLINKS=1 \
huggingface-cli download --repo-type dataset $repo_id --local-dir $hf_download_dir

end_time_hf_download=$(date +%s)
echo "Download completed. Time taken: $((end_time_hf_download - start_time_hf_download)) seconds"

# Step 2: Extract compressed array records in parallel
echo "Starting parallel extraction of tar files..."
num_workers=64  # Number of parallel extraction processes
start_time_uncompress=$(date +%s)

# Find all shard tar files and extract them in parallel:
xargs -0 -P $num_workers -I {} bash -c 'echo "Extracting {}"; tar -xf "{}" -C "'$final_dataset_dir'"'

end_time_uncompress=$(date +%s)

# Display timing summary
echo "================================"
echo "Extraction completed successfully!"
echo "Uncompress time: $((end_time_uncompress - start_time_uncompress)) seconds"
echo "Download time: $((end_time_hf_download - start_time_hf_download)) seconds"
echo "Total time: $((end_time_uncompress - start_time_hf_download)) seconds"
echo "Final dataset location: $final_dataset_dir"
