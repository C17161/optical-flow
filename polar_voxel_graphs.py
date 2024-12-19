# for each frame, plot timeseries / polar histogram for ONE voxel within the mask region
from utils import load_brain_data
brain_data, X, Y, masked_xyz = load_brain_data('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')

# imports
import matplotlib.pyplot as plt
import numpy as np
# for optical flow analysis
import cv2 as cv
import math

all_flow_x = []
start_frame = 0
end_frame = brain_data.shape[-1]


# specific voxel location within masked
y = masked_xyz[1][0]
z = masked_xyz[2][0]
# frame_flow_x.shape[0] 45
# frame_flow_x.shape[1] 96

voxel_angles_rad = []
voxel_angles_deg = []

# loop through all time stamps
# make sure the z and y aren't flipped!!!
for t in range(start_frame, end_frame - 1):
    # run farneback algorithm on greyscale images
    # structure: brain_data[X, Y, Z, time]
    current_gray_x = np.fliplr(brain_data[X,:,:,t]).T
    next_gray_x = np.fliplr(brain_data[X,:,:,t + 1]).T
    frame_flow_x = cv.calcOpticalFlowFarneback(current_gray_x, next_gray_x, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    fy, fz = frame_flow_x[z, y]  # extracts optical flow vector at a specific voxel (x, y)

    # calculate angle of the vector
    angle = np.arctan2(fy, fz)
    voxel_angles_rad.append(angle)
    angle = math.degrees(angle)
    voxel_angles_deg.append(angle)

# plot voxel on polar plot
fig, ax = plt.subplots(subplot_kw = dict(projection="polar"))
ax = ax.flatten()
ax.hist(voxel_angles_rad, bins = int(360/18), color = 'navy', edgecolor = 'white')
plt.title(f"Plot of Voxel {y}, {z} Angle (Radians) Over All Timestamps")
plt.show()

# plotting the voxel on a histogram
ax = plt.subplot(projection = 'polar')
plt.hist(voxel_angles_rad, bins = int(360/18), color = 'navy', edgecolor = 'white')
plt.title(f"Plot of Voxel {y}, {z} Angle (Radians) Over All Timestamps")
plt.show()

# # plotting the voxel over time stamps on a regular graph
# fig2, ax = plt.subplots()
# plt.plot(voxel_angles_deg)
# plt.title(f"Plot of Voxel {y}, {z} Angle (Degrees) Over All Timestamps {end_frame}")
# plt.show()




# NEXT STEPS 11/22
# polar histogram of ALL voxels in the mask
    # plot in subplot grid (9 by 9)
# look at different slices (Y slice)
    # repeat same visuals (make movie, graphs)
# idea: other optical flow algorithms & features
# make a video smoothing in time
    # for a time series, average data in window and smooth over the data
    # "moving average"
    # rapidly moving things should get removed, slowly moving trends should stay
    # "filter"



# ALL VOXELS IN Y MASKED REGION (8 total)
from utils import load_brain_data
brain_data, X, Y, masked_xyz = load_brain_data('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')

# imports
import matplotlib.pyplot as plt
import numpy as np
# for optical flow analysis
import cv2 as cv
import math

start_frame = 0
end_frame = brain_data.shape[-1]

# min & max zoomed in points
xmin = 38
xmax = 44
zmin = 45
zmax = 55

# specific voxel location within masked
x_masks = masked_xyz[0]
z_masks = masked_xyz[2]

voxel_angles_rad = []
voxel_angles_deg = []

all_flow_y = []
# loop through all time stamps
for t in range(start_frame, end_frame - 1):
    # run farneback algorithm on greyscale images
    # structure: brain_data[X, Y, Z, time]
    current_gray_y = np.fliplr(brain_data[:,Y,:,t]).T
    next_gray_y = np.fliplr(brain_data[:,Y,:,t + 1]).T

    frame_flow_y = cv.calcOpticalFlowFarneback(current_gray_y, next_gray_y, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    all_flow_y.append(frame_flow_y)

all_flow_y = np.asarray(all_flow_y)

# fy, fz = frame_flow_x[z, y]  # extracts optical flow vector at a specific voxel
all_angles = []
for y, z in zip(x_masks, z_masks):
    fy = all_flow_x[:,z,y,0]
    fz = all_flow_x[:,z,y,1]
    all_angles.append(np.arctan2(fy, fz))

# all_angles is a array of each of the 8 voxel's 579 angles 
fig, ax = plt.subplots(nrows=4,ncols=2, subplot_kw = dict(projection="polar"))
ax = ax.flatten()
for k in range(len(all_angles)):
    ax[k].hist(all_angles[k], bins = int(360/18), color = 'navy', edgecolor = 'white')
fig.suptitle(f"Plot of All Y-Axis Masked Voxel Angles (Degrees) Over All Timestamps")
plt.show()





















# ALL VOXELS IN X MASKED REGION (9 total)
# IN UTILS: all_voxel_degs_X
from utils import load_brain_data
brain_data, X, Y, masked_xyz = load_brain_data('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')

# imports
import matplotlib.pyplot as plt
import numpy as np
# for optical flow analysis
import cv2 as cv
import math

start_frame = 0
end_frame = brain_data.shape[-1]

# specific voxel location within masked
y_masks = masked_xyz[1]
z_masks = masked_xyz[2]
# frame_flow_x.shape[0] 45
# frame_flow_x.shape[1] 96

all_flow_x = []
# each timestamp's optical flow...
for t in range(start_frame, end_frame - 1):
    # run farneback algorithm on greyscale images
    # structure: brain_data[X, Y, Z, time]
    current_gray_x = np.fliplr(brain_data[X,:,:,t]).T
    next_gray_x = np.fliplr(brain_data[X,:,:,t + 1]).T
    frame_flow_x = cv.calcOpticalFlowFarneback(current_gray_x, next_gray_x, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    all_flow_x.append(frame_flow_x)

all_flow_x = np.asarray(all_flow_x)

# fy, fz = frame_flow_x[z, y]  # extracts optical flow vector at a specific voxel
all_angles = []
for y, z in zip(y_masks, z_masks):
    fy = all_flow_x[:,z,y,0]
    fz = all_flow_x[:,z,y,1]
    all_angles.append(np.arctan2(fy, fz))

# all_angles is a array of each of the 8 voxel's 579 angles 
fig, ax = plt.subplots(nrows=4,ncols=2, subplot_kw = dict(projection="polar"))
ax = ax.flatten()
for k in range(len(all_angles)):
    ax[k].hist(all_angles[k], bins = int(360/18), color = 'navy', edgecolor = 'white')
fig.suptitle(f"Plot of All X-Axis Masked Voxel Angles (Degrees) Over All Timestamps",
             fontsize = 25)
plt.show()


# RAYLEIGH TEST
from utils import load_brain_data
brain_data, X, Y, masked_xyz = load_brain_data('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')

from utils import all_voxel_degs_X_plot
all_angles = all_voxel_degs_X_plot(brain_data, masked_xyz, X)

# verify that shifting the x coordinates will get diff distributions
shift_val = 10
fake_mask = [np.array([50, 50, 50, 51, 51, 51, 51, 52]), np.array([63, 64, 64, 63, 63, 64, 64, 64]) + shift_val, np.array([2, 0, 1, 0, 2, 0, 1, 0])]
fake_angs = all_voxel_degs_X_plot(brain_data, fake_mask, X)

from astropy.stats import rayleightest
from astropy import units as u
import numpy as np
# for all the masked voxels
for i, angle_list in enumerate(all_angles):
    data = np.array(angle_list)*u.deg
    print(f'rayleigh test for voxel {i} is {rayleightest(data)}')




# to do: 11/25
# smoothing in time
# start to think about: moving from 2D to 3D 
    # not running 3D algorithms yet
    # "fake" do 3D by doing 2D analysis on slices next to each other
    # stack different x axis frames together
# compute seperately in x and y plane 
    # sub cube where the planes overlap
    # average them and see flow in 3D