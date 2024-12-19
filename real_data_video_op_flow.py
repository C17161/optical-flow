# saving an image for each optical flow timestamp
# OPTICAL FLOW OVER ALL FOR X SLICE
import numpy as np
import nibabel as nib
# from nilearn import plotting, image
import matplotlib.pyplot as plt
import numpy as np
# for optical flow analysis
import cv2 as cv
import matplotlib.patches as patches
import matplotlib.cm as cm # for color
import imageio.v2 as imageio
import os
os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
import sys
sys.path.append('/Users/catherinetu/Documents/Opticalflow')

from utils import load_brain_data
brain_data, X, Y, _ = load_brain_data('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')

# find the images over every time stamp
from utils import images_over_all_time
images_over_all_time('/Users/catherinetu/Documents/Opticalflow/output', brain_data, X)

# compiles all the images in a folder and saves as a video
from utils import create_video
images_dir = '/Users/catherinetu/Documents/Opticalflow/output'
video_dir = 'optical_flow_video.mp4'
create_video(image_folder = images_dir, output_video = video_dir, frame_rate = 10)


# find images over time steps of 3
from utils import images_over_steps_of_time
# look at every 3
time_step = 3
images_dir = '/Users/catherinetu/Documents/Opticalflow/steps_time'
video_dir = 'optical_flow_video_3steps.mp4'
images_over_steps_of_time(images_dir, brain_data, X, time_step)
create_video(images_dir, video_dir, 10)


# AVERAGED images over time steps of 3
from utils import images_over_steps_of_time_averaged
time_step = 3
images_dir_avg = '/Users/catherinetu/Documents/Opticalflow/average_steps_time'
video_dir_avg = 'avg_optical_flow_video_3steps.mp4'
images_over_steps_of_time_averaged(images_dir_avg, brain_data, X, time_step)
from utils import create_video
create_video(images_dir_avg, video_dir_avg, 10)







# USING FFMPEG DIRECTLY - NO WORK ):
# # with animation
# from matplotlib.animation import FFMpegWriter
# from imageio_ffmpeg import get_ffmpeg_exe
# import matplotlib as mpl

# ffmpeg_path = '/Users/catherinetu/Documents/Opticalflow/ffmpeg-7.1'
# # specify the ffmpeg executable 
# mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path

# # define writer
# writer = FFMpegWriter(
#     fps=2, 
#     metadata={'title': 'Optical Flow Video'}, 
#     extra_args=['-vcodec', 'libx264'], 
# )

# fig, ax = plt.subplots()
# # writer = FFMpegWriter(ffmpeg_path=ffmpeg_path)

# # loop through all time stamps
# with writer.saving(fig, '/Users/catherinetu/Documents/Opticalflow/output_movie.mp4', dpi=200):
#     for t in range(start_frame, end_frame - 1):
#         current_gray_x = np.fliplr(brain_data[X,:,:,t]).T
#         next_gray_x = np.fliplr(brain_data[X,:,:,t + 1]).T
#         frame_flow_x = cv.calcOpticalFlowFarneback(current_gray_x, next_gray_x, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#         step = 2
#         scalar = 30
#         ax.clear()
#         ax.imshow(next_gray_x, cmap='gray')

#         for y in range(0, frame_flow_x.shape[0], step):
#             for x in range(0, frame_flow_x.shape[1], step):
#                 fx, fy = frame_flow_x[y, x]
#                 end_point = (x + int(fx * scalar), y + int(fy * scalar))
#                 angle = np.arctan2(fy, fx)
#                 color_ang = cm.hsv((angle + np.pi) / (2 * np.pi))
#                 ax.arrow(x, y, fx * scalar, fy * scalar, color=color_ang, width=0.5)

#         writer.grab_frame()


# # IN TERMINAL
# # check perms: ls -l /Users/catherinetu/Documents/Opticalflow/ffmpeg
# # if file does not have x in it, it is not executable (accessable)
# # change perms: chmod +x /Users/catherinetu/Documents/Opticalflow/ffmpeg
# # chmod 777 /Users/catherinetu/Documents/Opticalflow
# # chmod +x /Users/catherinetu/Documents/Opticalflow/ffmpeg-7.1/configure
# # chmod -R +x /Users/catherinetu/Documents/Opticalflow/ffmpeg-7.1
# # ^ didn't work -- still had access denied error

# # tried reinstalling ffmpeg
# # pip install ffpeg-python
# # conda install ffmpeg
# # conda install pillow
# # conda install matplotlib

