# SMOOTH OVER TIME BY AVERAGING FLOW ESTIMATES OVER n TIME FRAMES
from utils import load_brain_data
brain_data, X, Y, masked_xyz = load_brain_data('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import matplotlib.cm as cm # for color

step = 2
scalar = 30

all_flow = []
start_frame = 0
end_frame = brain_data.shape[-1]

# loop through all time stamps
for t in range(start_frame, end_frame - 1):
    # run farneback algorithm on greyscale images
    # structure: brain_data[X, Y, Z, time]
    current_gray = np.fliplr(brain_data[X,:,:,t]).T
    next_gray = np.fliplr(brain_data[X,:,:,t + 1]).T

    frame_flow = cv.calcOpticalFlowFarneback(current_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    all_flow.append(frame_flow)

# get the average flow
mean_flow = np.mean(all_flow,axis=0)

# plot average flow 
fig, ax = plt.subplots()
ax.imshow(next_gray)

# Draw arrows on the ending frame
# step = spacing bt arrows in flow visual
# scalar = scale up arrow length
# iterating over flow field with step in both x and y
for y in range(0, mean_flow.shape[0], step):
    for x in range(0, mean_flow.shape[1], step):
        fx, fy = mean_flow[y,x]  # extracts optical flow vector at (x, y)
        end_point = (int(fx * scalar), int(fy * scalar))  # calc length of the arrow using flow vector
        # calculate angle of the vector for color coding
        angle = np.arctan2(fy, fx)
        # angle ranges from -pi to pi
            # Red: left movement.
            # Yellow/Green: up movement.
            # Cyan/Blue: right movement.
            # Purple/Magenta: down movement.
        # cm.hsv takes 0-1, which represents a full circle around color wheel
        color_ang = cm.hsv((angle + np.pi) / (2 * np.pi)) 
        # draw arrows from x, y to end_point
        arrow = patches.Arrow(x, y, dx=end_point[0], dy=end_point[1], width=2, color=color_ang)
        ax.add_patch(arrow)
plt.show()


# PROBLEM: gets rid of masked voxels, and does not average time stamps