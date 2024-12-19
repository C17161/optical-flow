# only need imports once per terminal
from nilearn import datasets
import numpy as np
# need this import to show the plot
import matplotlib.pyplot as plt
# this file contains a 3d volume -- visualize as statistical map
from nilearn import plotting, image
# fMRI DATASET
from nilearn.datasets import fetch_spm_auditory
from nilearn.plotting import plot_anat, plot_img
# load in functional image
import nibabel as nib
# get the mean image for a 4th dimension
import nilearn.image as nimg

motor_images = datasets.fetch_neurovault_motor_task()
# motor images = a list of file names -- we want the first one
# ['/Users/catherinetu/nilearn_data/neurovault/collection_658/image_10426.nii.gz']
motor_images.images
tmap_filename = motor_images.images[0]

plotting.plot_stat_map(tmap_filename)
plt.show()

# resting state networks
rsn = datasets.fetch_atlas_smith_2009(resting=True, dimension=10)["maps"]
rsn # '/Users/catherinetu/nilearn_data/smith_2009/PNAS_Smith09_rsn10.nii.gz'

print(image.load_img(rsn).shape) # prints it's shape: (91, 109, 91, 10)

# will get different volume images based on the index you choose (currently 0)
first_rsn = image.index_img(rsn, 0)
print(first_rsn.shape) # 3D image for plotting
plotting.plot_stat_map(first_rsn)
plt.show()

# iter_img on 4d images to print them
# threshold of 3.1 corresponds to p value of 0.01
for img in image.iter_img(rsn):
    plotting.plot_stat_map(img, threshold = 3.1, display_mode = 'z',
                           cut_coords = 1, colorbar = False)
plt.show() # displays the different images created (in reverse order)

selected_volumes = image.index_img(rsn, slice(3, 5))
for img in image.iter_img(selected_volumes):
    plotting.plot_stat_map(img)
plt.show()

subject_data = fetch_spm_auditory()
print(*subject_data.func[:5], sep="\n")  # print paths of first 5 func images

plot_img(subject_data.func[0], colorbar=True, cbar_tick_format="%i")
plt.show()
plot_anat(subject_data.anat, colorbar=True, cbar_tick_format="%i")
plt.show()

# reformat the data to be 4D
img_list = []
for k in range(80):
    functional_img = nib.load(subject_data.func[k])

    f_data = functional_img.get_fdata()
    # load the fMRI data in list
    img_list.append(f_data)
    if k % 20 ==0:
        print(k)
# convert fmri list to a numpy array
img_array = np.asarray(img_list)
img_array = np.moveaxis(img_array, 0, -1)
# print(img_array)
# BRAIN DATA PUT INTO COMPATIBLE NIFTI DATA!
img_obj = nib.Nifti1Image(img_array, functional_img.affine, functional_img.header)

# reorder array so that timestamp is at the back
# dimensions = image.load_img(img_obj).shape
# print(dimensions)
# time_stamps = dimensions[0]

# plot the mean image
mean_image = nimg.mean_img(image.load_img(img_obj))
print(mean_image.shape)
plotting.plot_stat_map(mean_image,bg_img=None)
plt.show()

# BEFORE FRIDAY OCT 25
# 1- average spatially and plot a time series of signal over time


# average across the spatial dimensions (x, y, z) for each time point
# reduces the 4D array to a 1D array with the time series
time_series = np.mean(img_array, axis=(0, 1, 2))

# Plot the time series of the signal over time
plt.plot(time_series, color = 'magenta')
plt.title("Time Series of fMRI Signal")
plt.xlabel("Time (scans)")
plt.ylabel("Mean Signal Intensity")
plt.show()

# 1.5 - average spatially and plot a time series of left brain & right brain
# signals over time

# Load img obj as NIfTI image
nifti_data = img_obj.get_fdata()
# print(nifti_data)

midline = nifti_data.shape[0] // 2

# extract different brain signals
left_brain_signal = nifti_data[:midline, :, :,:]
right_brain_signal = nifti_data[midline:, :, :,:]

# calculate average signals for each hemisphere
avg_left_sig = np.mean(left_brain_signal, axis = (0, 1, 2))
avg_right_sig = np.mean(right_brain_signal, axis = (0, 1, 2))

plt.imshow(np.squeeze([nifti_data[30,:,:,10]]));plt.show()


# plot left and right brain signals on the same graph
plt.plot(avg_left_sig, color = 'turquoise', label = 'Left Side')
plt.plot(avg_right_sig, color = 'blue', label = 'Right Side')
plt.title("Time Series of Left & Right Brain fMRI Signal")
plt.xlabel("Time (scans)")
plt.ylabel("Mean Signal Intensity")
plt.legend()
plt.show()

# 2 - additional plot on it with the brain: single data from one time on one half and the average
# of all images on other side seperately and combine them together half and half

def half_and_half(n, axis):
    '''
    given time point n and an axis (0 = x, 1 = y, 2 = z), plot the left half of the brain
    as what the brain looked like at time point n and the right half as an average along
    the chosen axis
    '''
    axis_text = {0: 'x', 1: 'y', 2: 'z'}
    # load time point data n chosen by user
    single_time_img = img_array[..., n]  # Select the 0th time point (or any other)

    # load the average image
    average_img = np.mean(img_array, axis=-1)  # Average over time

    # create a half-and-half combination of the two images
    # split along the middle of the brain (let's assume it's the x-axis, you can adjust)
    mid_point = single_time_img.shape[axis] // 2

    # create a new combined image, where the left half is from 'single_time_img' and the right half from 'average_img'
    combined_img = np.zeros_like(single_time_img)
    if axis == 0:  # split along the x-axis
        combined_img[:mid_point, :, :] = single_time_img[:mid_point, :, :]
        combined_img[mid_point:, :, :] = average_img[mid_point:, :, :]
    elif axis == 1:  # split along the y-axis
        combined_img[:, :mid_point, :] = single_time_img[:, :mid_point, :]
        combined_img[:, mid_point:, :] = average_img[:, mid_point:, :]
    elif axis == 2:  # split along the z-axis
        combined_img[:, :, :mid_point] = single_time_img[:, :, :mid_point]
        combined_img[:, :, mid_point:] = average_img[:, :, mid_point:]

    # convert the combined image back to a NIfTI image object
    combined_nifti_img = nib.Nifti1Image(combined_img, functional_img.affine, functional_img.header)

    # Step 5: Plot the combined brain image
    plotting.plot_stat_map(combined_nifti_img, title = f'single time point {n} vs average image along {axis_text[axis]} axis',
                           display_mode='ortho', bg_img=None)
    plt.show()
# time point data n on the single brain data point
half_and_half(0, 0)
half_and_half(5, 0)
half_and_half(5, 2)