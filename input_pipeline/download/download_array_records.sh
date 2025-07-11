# Download array records from Hugging Face
# 1. Download compressed array records
# 2. Uncompress array records

hf_download_dir='data/minecraft_arrayrecords_compressed' # destination for hf download of compressed dataset
final_dataset_dir='data/minecraft_arrayrecords'          # destination for final uncompressed array records

mkdir -p $hf_download_dir
mkdir -p $final_dataset_dir

# Download compressed dataset from Hugging Face
repo_id=avocadoali/open_ai_minecraft_arrayrecords_chunked
start_time_hf_download=$(date +%s)
HF_HUB_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_SYMLINKS=1 \
huggingface-cli download --repo-type dataset $repo_id --local-dir $hf_download_dir
end_time_hf_download=$(date +%s)
echo "Download done. Time taken: $((end_time_hf_download - start_time_hf_download)) seconds"

# Uncompress array records
num_workers=62
start_time_uncompress=$(date +%s)
find $hf_download_dir -name "shard_*.tar" -print0 | \
xargs -0 -P $num_workers -I {} bash -c 'echo "Extracting {}"; tar -xf "{}" -C "'$final_dataset_dir'"'
end_time_uncompress=$(date +%s)
echo "--------------------------------"
echo "Uncompress time: $((end_time_uncompress - start_time_uncompress)) seconds"
echo "Download time: $((end_time_hf_download - start_time_hf_download)) seconds"
echo "Total time: $((end_time_uncompress - start_time_hf_download)) seconds"
