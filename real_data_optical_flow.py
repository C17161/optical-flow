# TIME SERIES OVER VOXELS IN MASK ----------
# import os
import numpy as np
import nibabel as nib
# from nilearn import plotting, image
import matplotlib.pyplot as plt
import numpy as np
# for optical flow analysis
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.cm as cm # for color

# brain
brain = nib.load('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')
brain_data = brain.get_fdata()
affine = brain.affine

# mask
mask = nib.load('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_desc-V4_mask.nii')
mask_data = mask.get_fdata()
mask_indexes = (mask_data == 1) # look for all data with 1 to denote mask

# checks
print(brain_data.shape) # (96, 96, 45, 580)
print(mask_data.shape) # (96, 96, 45)

# avg the 4D brain data along the time dimension to get a 3D image
brain_data_avg = brain_data.mean(axis = -1)

# convert to nib nifti image
brain_avg_img = nib.Nifti1Image(brain_data_avg, affine)

# find indices of the mask voxels 
[x,y,z] = np.where(mask_indexes)

X = int(np.mean(x))
Y = int(np.mean(y))
Z = int(np.mean(z))

#apply mask to the data and plot timeseries of each voxel in the mask
plt.plot(brain_data[mask_data==1].T)
plt.title("Time Series of Masked Voxels")
plt.xlabel("Time")
plt.ylabel("Signal Intensity")
plt.show()

# apply the mask
brain_data_masked = np.zeros_like(brain_data_avg)
brain_data_masked[mask_data == 1] = brain_data_avg[mask_data == 1]

# masked_brain_data = brain_data[mask_indexes]
brain_data_masked_img = nib.Nifti1Image(brain_data_masked, affine)
# nib.save(brain_data_masked_img, '/Users/catherinetu/Documents/Opticalflow/test')

print(brain_data_masked_img.shape) # (96, 96, 45)

# plot_anat is for visualizing 2D cross section slices of 3D/4D data
# plotting.plot_anat(brain_avg_img, title = "Masked Brain Image", display_mode = "ortho",cut_coords=[X,Y,Z])
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=[10,5])
ax[0].imshow(np.fliplr(brain_data_avg[X,:,:]).T)
ax[1].imshow(np.fliplr(brain_data_avg[:,Y,:]).T)
ax[2].imshow(brain_data_avg[:,:,Z].T)
plt.show()

# test farneback:
# brain_data[X,:,:,3]
# cv.calcOpticalFlowFarneback(brain_data[X,:,:,2], brain_data[X,:,:,3], None, 0.5, 3, 15, 3, 5, 1.2, 0)






# APPLYING OPTICAL FLOW ALGORITHM BETWEEN 2 TIME POINTS OF BRAIN DATA --------


# calculate dense optical flow between the two frames
# import os
import numpy as np
import nibabel as nib
# from nilearn import plotting, image
import matplotlib.pyplot as plt
import numpy as np
# for optical flow analysis
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.cm as cm # for color

# brain
brain = nib.load('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')
brain_data = brain.get_fdata()
affine = brain.affine

# mask
mask = nib.load('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_desc-V4_mask.nii')
mask_data = mask.get_fdata()
mask_indexes = (mask_data == 1) # look for all data with 1 to denote mask

# find indices of the mask voxels 
[x,y,z] = np.where(mask_indexes)

X = int(np.mean(x))

all_flow = []
start_frame = 0
end_frame = 2

current_gray = np.fliplr(brain_data[X,:,:,3]).T
next_gray = np.fliplr(brain_data[X,:,:,4]).T

frame_flow = cv.calcOpticalFlowFarneback(current_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# append a transposed version that is correct orientation
# frame_flow_T = np.fliplr(frame_flow).T
all_flow.append(frame_flow)
# image we are graphing

frame_flow.shape
# frame_flow_T.shape

# for t in range(start_frame, end_frame):
#     # run farneback algorithm on greyscale images
#     # structure: brain_data[X, Y, Z, time]
#     current_gray = cv.cvtColor(brain_data[X,:,:,t], cv.COLOR_BGR2GRAY)
#     next_gray = cv.cvtColor(brain_data[X,:,:,t + 1], cv.COLOR_BGR2GRAY)

#     # calculate dense optical flow between the two frames
#     frame_flow = cv.calcOpticalFlowFarneback(current_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     all_flow.append(frame_flow)
#     last_frame = brain_data[X,:,:,t + 1]

# plot average flow 
fig, ax = plt.subplots()
ax.imshow(current_gray)

# Draw arrows on the ending frame
step = 3  #spacing bt arrows in flow visual
scalar = 30
# iterating over flow field with step in both x and y
for y in range(0, frame_flow.shape[0], step):
    for x in range(0, frame_flow.shape[1], step):
        fx, fy = frame_flow[y,x]  # extracts optical flow vector at (x, y)
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
        arrow = patches.Arrow(x, y, dx=end_point[0], dy=end_point[1], width=3, color=color_ang)
        print(arrow)
        ax.add_patch(arrow)
# fig.set_title("Optical Flow Vectors")
# plt.gca().invert_yaxis()
# ax.transAxes
plt.show()






# OPTICAL FLOW OVER ALL FOR X SLICE
import numpy as np
import nibabel as nib
# from nilearn import plotting, image
import matplotlib.pyplot as plt
import numpy as np
# for optical flow analysis
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.cm as cm # for color

from utils import load_brain_data
brain_data, X, Y = load_brain_data('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')

# plt mean optical flow over X slice 
from utils import mean_optical_flow_X_slice
mean_optical_flow_X_slice(brain_data, X, 2, 30)











# OPTICAL FLOW OVER ALL FOR X AND Y SLICE
import numpy as np
import nibabel as nib
# from nilearn import plotting, image
import matplotlib.pyplot as plt
import numpy as np
# for optical flow analysis
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.cm as cm # for color

# brain
brain = nib.load('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')
brain_data = brain.get_fdata()
affine = brain.affine
# brain_data.shape

# mask
mask = nib.load('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_desc-V4_mask.nii')
mask_data = mask.get_fdata()
mask_indexes = (mask_data == 1) # look for all data with 1 to denote mask

# find indices of the mask voxels in X slice
[x,y,z] = np.where(mask_indexes)
X = int(np.mean(x))
Y = int(np.mean(y))
Z = int(np.mean(z))

# plt.imshow(brain_data[:,Y+1,:,10]);plt.show()

all_flow_x = []
all_flow_y = []
all_flow_z = []
start_frame = 0
end_frame = brain_data.shape[-1]

# loop through all time stamps
for t in range(start_frame, end_frame - 1):
    # run farneback algorithm on greyscale images
    # structure: brain_data[X, Y, Z, time]
    current_gray_x = np.fliplr(brain_data[X,:,:,t]).T
    next_gray_x = np.fliplr(brain_data[X,:,:,t + 1]).T
    current_gray_y = np.fliplr(brain_data[:,Y,:,t]).T
    next_gray_y = np.fliplr(brain_data[:,Y,:,t + 1]).T
    current_gray_z = brain_data[:,:,Z,t]
    next_gray_z = brain_data[:,:,Z,t + 1]

    frame_flow_x = cv.calcOpticalFlowFarneback(current_gray_x, next_gray_x, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    all_flow_x.append(frame_flow_x)

    frame_flow_y = cv.calcOpticalFlowFarneback(current_gray_y, next_gray_y, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    all_flow_y.append(frame_flow_y)

    frame_flow_z = cv.calcOpticalFlowFarneback(current_gray_z, next_gray_z, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    all_flow_z.append(frame_flow_z)

# print(frame_flow_x.shape)
# print(frame_flow_y.shape)

# get the average flow
mean_flow_x = np.mean(all_flow_x,axis=0)
mean_flow_y = np.mean(all_flow_y,axis=0)
mean_flow_z = np.mean(all_flow_z,axis=0)

# plot average flow 
# fig, ax = plt.subplots(nrows=1,ncols=3,figsize=[10,5])
# ax[0].imshow(next_gray_x)
# ax[1].imshow(next_gray_y)
# ax[2].imshow(next_gray_z)


# Draw arrows on the ending frame
step = 2  #spacing bt arrows in flow visual
scalar = 30
# iterating over flow field for axis X
fig, ax = plt.subplots()
ax.imshow(next_gray_x)

for y in range(0, mean_flow_x.shape[0], step):
    for x in range(0, mean_flow_x.shape[1], step):
        fx, fy = mean_flow_x[y,x]  # extracts optical flow vector at (x, y)
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
        arrow = patches.Arrow(x, y, dx=end_point[0], dy=end_point[1], width=3, color=color_ang)
        ax.add_patch(arrow)
plt.show()


# for axis y
fig, ax = plt.subplots()
ax.imshow(next_gray_y)

for y in range(0, mean_flow_y.shape[0], step):
    for x in range(0, mean_flow_y.shape[1], step):
        fx, fy = mean_flow_y[y,x]  # extracts optical flow vector at (x, y)
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
        arrow = patches.Arrow(x, y, dx=end_point[0], dy=end_point[1], width=3, color=color_ang)
        ax.add_patch(arrow)
plt.show()

# for axis z
fig, ax = plt.subplots()
ax.imshow(next_gray_z)
#ymin,ymax
#xmin,xmax
#imshow(mean_flow_z[ymin:ymax,xmin:xmax])
#arrow(x-xmin,y-ymin)
for y in range(0, mean_flow_z.shape[0], step):
    for x in range(0, mean_flow_z.shape[1], step):
        fx, fy = mean_flow_z[y,x]  # extracts optical flow vector at (x, y)
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
        arrow = patches.Arrow(x, y, dx=end_point[0], dy=end_point[1], width=3, color=color_ang)
        ax.add_patch(arrow)
plt.show()



# TO DO 11/15/24
# 1 - zoom in on the ventricle area and plot the average flow over time (blow it up)
    # compare images zoomed in vs zoomed
    # for loop from min to max
    #ymin,ymax
    #xmin,xmax
    #imshow(mean_flow_z[ymin:ymax,xmin:xmax])
    #arrow(x-xmin,y-ymin)
# 2 - make a video out of the different plot photos (vector field)

# improvements to mean vector flow:
# flow moving up and down - average to 0
# a bunch of things that the mean doesnt help us -- many scenarios
# some way to find where the flow vectors are in the data

# improvement 2: this is just one subject
# check another subject and check if the data works with other people




# make sure dir exists: mkdir -p /Users/catherinetu/Documents/Opticalflow
# allow write access: chmod u+w /Users/catherinetu/Documents/Opticalflow
# write access to all users: chmod 777 /Users/catherinetu/Documents/Opticalflow
# check perms: ls -ld /Users/catherinetu/Documents/Opticalflow
# drwxrwxrwx  14 catherinetu  staff  448 Nov 20 10:44 /Users/catherinetu/Documents/Opticalflow
# enables perms to dir




# ZOOM IN ON VENTRICLE AREA - X AXIS
# OPTICAL FLOW OVER X SLICE

from utils import load_brain_data
brain_data, X, Y, _ = load_brain_data('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')

# plot the zoomed in X slice over the ventricle
from utils import zoomed_in_optical_flow_X
zoomed_in_optical_flow_X(brain_data, X, 40, 50, 50, 80)


# CONTROL COMPARISON: shift the zoomed in data off the ventricle
zoomed_in_optical_flow_X(brain_data, X, 40, 50, 70, 100)







# FIND ZOOMED IN AREA ON THE Y AXIS
from utils import load_brain_data
brain_data, X, Y, _ = load_brain_data('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_bold_MC.nii')

start_frame = 0
end_frame = brain_data.shape[-1]

all_flow_y = []
# loop through all time stamps
for t in range(start_frame, end_frame - 1):
    # run farneback algorithm on greyscale images
    # structure: brain_data[X, Y, Z, time]
    current_gray_y = np.fliplr(brain_data[:,Y,:,t]).T
    next_gray_y = np.fliplr(brain_data[:,Y,:,t + 1]).T

    frame_flow_y = cv.calcOpticalFlowFarneback(current_gray_y, next_gray_y, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    all_flow_y.append(frame_flow_y)

# get the average flow
mean_flow_y = np.mean(all_flow_y, axis=0)

# min & max zoomed in points
xmin = 38
xmax = 44
ymin = 45
ymax = 55

# Draw arrows on the ending frame
step = 2  #spacing bt arrows in flow visual
scalar = 30
# iterating over flow field for axis X
fig, ax = plt.subplots()
ax.imshow(next_gray_y[xmin:xmax + 1, ymin:ymax + 1])

for y in range(0, mean_flow_y.shape[0], step):
    for x in range(0, mean_flow_y.shape[1], step):
        fx, fy = mean_flow_y[y,x]  # extracts optical flow vector at (x, y)
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
        arrow = patches.Arrow(x-xmin, y-ymin, dx=end_point[0], dy=end_point[1], width=3, color=color_ang)
        ax.add_patch(arrow)
fig.suptitle("Zoomed In On Ventricle: Average Optical Flow")
plt.show()






# TEST ON DIFFERENT DATASET (different state, same person)
import nibabel as nib
import numpy as np

pre_brain_dir = '/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-pre_acq-csf_bold_MC.nii'
brain = nib.load(pre_brain_dir)
brain_data = brain.get_fdata()
affine = brain.affine
# brain_data.shape

# mask
mask = nib.load('/Users/catherinetu/Documents/Opticalflow/sub-dex7t03_ses-1_task-dex_acq-csf_desc-V4_mask.nii')
mask_data = mask.get_fdata()
mask_indexes = (mask_data == 1) # look for all data with 1 to denote mask

# find indices of the mask voxels in X slice
masked_xyz = np.where(mask_indexes)
X = int(np.mean(x))
Y = int(np.mean(y))

# mean flow
from utils import mean_optical_flow_X_slice
mean_optical_flow_X_slice(brain_data, X, 2, 30)

pre_dir = '/Users/catherinetu/Documents/Opticalflow/pre_csf'

# images of flow
from utils import images_over_all_time
images_over_all_time(pre_dir, brain_data, X)

# create video from photos
from utils import create_video
create_video(pre_dir, 'pre_all_frames_video.mp4', frame_rate=30)

# create smoothed images
pre_dir_steps = '/Users/catherinetu/Documents/Opticalflow/pre_csf_3steps'
from utils import images_over_steps_of_time_averaged
images_over_steps_of_time_averaged(pre_dir_steps, brain_data, X, 3)

# create video from smoothed photos
create_video(pre_dir_steps, 'pre_smoothed3_frames_video.mp4', frame_rate=30)

# voxel analysis
from utils import all_voxel_degs_X_plot
all_voxel_degs_X_plot(brain_data, masked_xyz, X)