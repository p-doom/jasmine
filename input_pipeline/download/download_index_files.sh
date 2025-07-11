# Download index files from OpenAI Video-Pre-Training dataset
# https://github.com/openai/Video-Pre-Training

output_dir="data/open_ai_index_files" # destination for index files

mkdir -p $output_dir

index_files=(
    "all_6xx_Jun_29.json"
    "all_7xx_Apr_6.json"
    "all_8xx_Jun_29.json"
    "all_9xx_Jun_29.json"
    "all_10xx_Jun_29.json"
)

for index_file in "${index_files[@]}"; do
    wget https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/$index_file -O $output_dir/$index_file
done
