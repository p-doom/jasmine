import json
import requests
import os
import tyro
import logging
from urllib.parse import urljoin
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time


@dataclass
class DownloadVideos:
    index_file_path: str = "data/open_ai_index_files/all_6xx_Jun_29.json"
    num_workers: int = -1
    output_dir: str = "data/minecraft_videos/"


def download_single_file(args):
    """Download a single file - designed to be used with multiprocessing"""
    relpath, url, output_path = args

    if os.path.exists(output_path):
        return f"Skipped {relpath} (already exists)"

    # No need to create parent directories since we're flattening the structure
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            file_size = 0
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        file_size += len(chunk)

            # Convert to MB for logging
            file_size_mb = file_size / (1024 * 1024)
            return f"Downloaded {relpath} ({file_size_mb:.2f} MB)"
        else:
            return f"Failed to download {relpath}: HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Request failed for {relpath}: {e}"
    except Exception as e:
        return f"Unexpected error downloading {relpath}: {e}"


def flatten_path(relpath):
    """Convert nested path to flattened filename with subdirectory as prefix
    e.g. data/6.10/filename.mp4 -> 6.10_filename.mp4
    """

    parts = relpath.split("/")

    if len(parts) >= 3:
        subdir = parts[1]
        filename = parts[2]
        return f"{subdir}_{filename}"
    else:
        return relpath.replace("/", "_")


def download_dataset(index_file_path, output_dir, num_workers=64):
    # Load the index file
    with open(index_file_path, "r") as f:
        index_data = json.load(f)

    basedir = index_data["basedir"]
    relpaths = index_data["relpaths"]

    # Filter for mp4 files only and flatten the path structure
    mp4_files = []
    for relpath in relpaths:
        if relpath.endswith(".mp4"):
            url = urljoin(basedir, relpath)
            flattened_filename = flatten_path(relpath)
            output_path = os.path.join(output_dir, flattened_filename)
            mp4_files.append((relpath, url, output_path))

    print(f"Found {len(mp4_files)} MP4 files to download")
    print(f"Using {num_workers} workers for parallel downloads")

    start_time = time.time()

    if num_workers > len(mp4_files):
        num_workers = len(mp4_files)

    with tqdm(
        total=len(mp4_files), desc="Overall Download Progress", unit="files"
    ) as pbar:
        with Pool(processes=num_workers) as pool:
            results = []
            for result in pool.imap_unordered(
                download_single_file,
                [
                    (relpath, url, output_path)
                    for relpath, url, output_path in mp4_files
                ],
            ):
                results.append(result)
                pbar.update(1)
    # Print final results summary
    successful_downloads = sum(1 for r in results if "Downloaded" in r)
    skipped_files = sum(1 for r in results if "Skipped" in r)
    failed_downloads = len(results) - successful_downloads - skipped_files

    print(f"\nDownload Summary:")
    print(f"  Successful downloads: {successful_downloads}")
    print(f"  Skipped files: {skipped_files}")
    print(f"  Failed downloads: {failed_downloads}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Download completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    args = tyro.cli(DownloadVideos)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_workers == -1:
        args.num_workers = cpu_count()

    print(f"Index file path: {args.index_file_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of workers: {args.num_workers}")

    download_dataset(args.index_file_path, args.output_dir, args.num_workers)
