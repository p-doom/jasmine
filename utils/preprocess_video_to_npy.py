import ffmpeg
import numpy as np
import os
import tyro
import multiprocessing as mp
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

@dataclass
class Args:
    target_width, target_height = 160, 90
    target_fps = 10
    input_path: str = "/hkfs/work/workspace/scratch/tum_ind3695-jafa_ws_shared/data/knoms/"
    output_path: str = "/hkfs/work/workspace/scratch/tum_ind3695-jafa_ws_shared/data/knoms_npy_tmp"


# Preprocess one video and save it to a .npy file
def preprocess_video(in_filename, output_path, target_width, target_height, target_fps):

    # Stream the video as raw RGB24, resize, and set fps
    out, _ = (
        ffmpeg
        .input(in_filename)
        .filter('fps', fps=target_fps, round='up')
        .filter('scale', target_width, target_height)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=False)
    )

    frame_size = target_height * target_width * 3
    n_frames = len(out) // frame_size

    frames = np.frombuffer(out, np.uint8).reshape(n_frames, target_height, target_width, 3)

    # Create output directory if it doesn't exist
    output_file = f'{output_path}/{in_filename.split("/")[-1].split(".")[0]}.npy'
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Save as .npy
    np.save(output_file, frames)
    print(f"Saved {n_frames} frames to {output_file} with shape {frames.shape}")


def get_n_from_file(filename, directory):
    filepath = os.path.join(directory, filename)
    arr = np.load(filepath, mmap_mode='r')  
    return filepath, arr.shape[0]


def main():
    args = tyro.cli(Args)

    # creates a list of arguments for the preprocess_video function
    print("Output path: ", output_path)
    output_path = f'{args.output_path}/{args.target_fps}fps_{args.target_width}x{args.target_height}'

    print("Starting to process videos...")
    filenames = [args.input_path + filename for filename in os.listdir(args.input_path) if filename.endswith('.mp4') or filename.endswith('.webm')]
    output_paths = [output_path] * len(filenames)
    target_widths = [args.target_width] * len(filenames)
    target_heights = [args.target_height] * len(filenames)
    target_fpss = [args.target_fps] * len(filenames)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(preprocess_video, filenames, output_paths, target_widths, target_heights, target_fpss), total=len(filenames)))

    # Create metadata file 
    metadata = []
    npy_files = [f for f in os.listdir(output_path) if f.endswith('.npy') and f != 'metadata.npy']
    print("Creating metadata file...")
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(get_n_from_file, npy_files, [output_path] * len(npy_files)), total=len(npy_files)))
        metadata = [{"path": path, "length": length} for path, length in results]
    np.save(output_path + '/metadata.npy', metadata)

    print("Done")
   

if __name__ == "__main__":
    main()
