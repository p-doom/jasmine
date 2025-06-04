import ffmpeg
import numpy as np

# Input video
# in_filename = '/home/hk-project-pai00039/tum_ind3695/projects/jafar/data/minecraft_videos/mc_gp_low_res_short.mp4'
in_filename = '/home/hk-project-pai00039/tum_ind3695/projects/jafar/data/minecraft_videos/mc_gp_low_res_long.mp4'

# Desired output properties
target_width, target_height = 160, 90
target_fps = 10

# Use ffprobe to get original video info (optional, for reference)
p = ffmpeg.probe(in_filename, select_streams='v')
orig_width = int(p['streams'][0]['width'])
orig_height = int(p['streams'][0]['height'])
orig_fps = eval(p['streams'][0]['r_frame_rate'])
print(f"Original: {orig_width}x{orig_height} @ {orig_fps} fps")

# Stream the video as raw RGB24, resize, and set fps
out, _ = (
    ffmpeg
    .input(in_filename)
    .filter('fps', fps=target_fps, round='up')
    .filter('scale', target_width, target_height)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run(capture_stdout=True)
)

frame_size = target_height * target_width * 3
n_frames = len(out) // frame_size

frames = np.frombuffer(out, np.uint8).reshape(n_frames, target_height, target_width, 3)

# Save as .npy
output_file = 'data/minecraft_py/mc_gp_low_res_long_10fps_160x90.npy'
np.save(output_file, frames)
print(f"Saved {n_frames} frames to {output_file} with shape {frames.shape}")

