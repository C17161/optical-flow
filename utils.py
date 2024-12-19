import nibabel as nib
import numpy as np

def load_brain_data(brain_dir):
    '''
    loads the brain data and mask given a certain directory
    '''
    brain = nib.load(brain_dir)
    brain_data = brain.get_fdata()
    affine = brain.affine
    # brain_data.shape

    # mask
    mask = nib.load(brain_dir.replace('bold_MC', 'desc-V4_mask'))
    mask_data = mask.get_fdata()
    mask_indexes = (mask_data == 1) # look for all data with 1 to denote mask

    # find indices of the mask voxels in X slice
    [x,y,z] = np.where(mask_indexes)
    X = int(np.mean(x))
    Y = int(np.mean(y))

    return brain_data, X, Y, [x, y, z]



import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import matplotlib.cm as cm # for color

def mean_optical_flow_X_slice(brain_data, X, step, scalar):
    '''
    plots the mean optical flow over all the timestamps in the X slice
    '''
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



def zoomed_in_optical_flow_X(brain_data, X, ymin, ymax, xmin, xmax):
    '''
    given certain min / max regions, plots a zoomed in mean optical flow
    plot over that region
    '''
    all_flow_x = []
    start_frame = 0
    end_frame = brain_data.shape[-1]

    # loop through all time stamps
    for t in range(start_frame, end_frame - 1):
        # run farneback algorithm on greyscale images
        # structure: brain_data[X, Y, Z, time]
        current_gray_x = np.fliplr(brain_data[X,:,:,t]).T
        next_gray_x = np.fliplr(brain_data[X,:,:,t + 1]).T

        frame_flow_x = cv.calcOpticalFlowFarneback(current_gray_x, next_gray_x, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        all_flow_x.append(frame_flow_x)

    # get the average flow
    mean_flow_x = np.mean(all_flow_x,axis=0)

    # Draw arrows on the ending frame
    step = 2  #spacing bt arrows in flow visual
    scalar = 30
    # iterating over flow field for axis X
    fig, ax = plt.subplots()
    ax.imshow(next_gray_x[ymin:ymax + 1, xmin:xmax + 1])
    #imshow(mean_flow_z[ymin:ymax,xmin:xmax])
        #arrow(x-xmin,y-ymin)

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
            arrow = patches.Arrow(x-xmin, y-ymin, dx=end_point[0], dy=end_point[1], width=3, color=color_ang)
            ax.add_patch(arrow)
    fig.suptitle("Zoomed In X Slice: Average Optical Flow")
    plt.show()


import imageio
import os
os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
import sys
sys.path.append('/Users/catherinetu/Documents/Opticalflow')


def images_over_all_time(folder_dir, brain_data, X):
    '''
    given brain data and a desired folder directory,
    print images of optical flow over the brain data
    over all time stamps
    '''
    # movie img dir 
    datadir = folder_dir
    end_frame = brain_data.shape[-1]


    # loop through all time stamps
    for t in range(0,  end_frame-1):
        print(f"Processing frame {t}...")

        # run farneback algorithm on greyscale images
        # structure: brain_data[X, Y, Z, time]
        current_gray_x = np.fliplr(brain_data[X,:,:,t]).T
        next_gray_x = np.fliplr(brain_data[X,:,:,t + 1]).T
        frame_flow_x = cv.calcOpticalFlowFarneback(current_gray_x, next_gray_x, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # draw arrows on the ending frame
        step = 2  #spacing bt arrows in flow visual
        scalar = 30
        # iterating over flow field for axis X
        fig, ax = plt.subplots()
        ax.imshow(next_gray_x)
        for y in range(0, frame_flow_x.shape[0], step):
            for x in range(0, frame_flow_x.shape[1], step):
                fx, fy = frame_flow_x[y,x]  # extracts optical flow vector at (x, y)
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

        fig.suptitle("Optical Flow Timestamp {}".format(t))
        # create file name to save temporary image
        fname = os.path.join(datadir,'x_axis_img{}.png'.format(t))
        plt.savefig(fname,dpi=200) # save image 
        plt.close() # close figure 

    print(f"photos saved to {folder_dir}")


def images_over_steps_of_time(folder_dir, brain_data, X, time_step):
    '''
    given brain data and a desired folder directory,
    print images of optical flow over the brain data
    over all time stamps
    '''
    # movie img dir 
    datadir = folder_dir
    end_frame = brain_data.shape[-1]

    # loop through all time stamps
    for t in range(0, end_frame-1, time_step):
        print(f"Processing frame {t}...")

        # run farneback algorithm on greyscale images
        # structure: brain_data[X, Y, Z, time]
        current_gray_x = np.fliplr(brain_data[X,:,:,t]).T
        next_gray_x = np.fliplr(brain_data[X,:,:,t + time_step]).T
        frame_flow_x = cv.calcOpticalFlowFarneback(current_gray_x, next_gray_x, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # draw arrows on the ending frame
        step = 2  #spacing bt arrows in flow visual
        scalar = 30
        # iterating over flow field for axis X
        fig, ax = plt.subplots()
        ax.imshow(next_gray_x)
        for y in range(0, frame_flow_x.shape[0], step):
            for x in range(0, frame_flow_x.shape[1], step):
                fx, fy = frame_flow_x[y,x]  # extracts optical flow vector at (x, y)
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

        fig.suptitle("Optical Flow Timestamp {}".format(t + time_step))
        # create file name to save temporary image
        fname = os.path.join(datadir,'x_axis_img{}.png'.format(t + time_step))
        plt.savefig(fname,dpi=200) # save image 
        plt.close() # close figure 

    print(f"photos saved to {folder_dir}")


def images_over_steps_of_time_averaged(folder_dir, brain_data, X, time_step):
    '''
    given brain data and a desired folder directory,
    print images of optical flow over the brain data
    over all time stamps
    '''
    # movie img dir 
    datadir = folder_dir
    end_frame = brain_data.shape[-1]

    # loop through all time stamps
    for t in range(0, end_frame-1, time_step):
        print(f"Processing frame {t}...")

        time_step_avg_flow = []
        for i in range(t, t + time_step):
            # run farneback algorithm on greyscale images
            # structure: brain_data[X, Y, Z, time]
            current_gray_x = np.fliplr(brain_data[X,:,:,i]).T
            next_gray_x = np.fliplr(brain_data[X,:,:,i + 1]).T
            frame_flow_x = cv.calcOpticalFlowFarneback(current_gray_x, next_gray_x, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            time_step_avg_flow.append(frame_flow_x)
        # find mean flow
        mean_flow = np.mean(time_step_avg_flow, axis=0)

        # draw arrows on the ending frame
        step = 2  #spacing bt arrows in flow visual
        scalar = 30
        # iterating over flow field for axis X
        fig, ax = plt.subplots()
        ax.imshow(next_gray_x)

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
                arrow = patches.Arrow(x, y, dx=end_point[0], dy=end_point[1], width=3, color=color_ang)
                ax.add_patch(arrow)

        fig.suptitle("Optical Flow Timestamp {}".format(t + time_step))
        # create file name to save temporary image
        fname = os.path.join(datadir,'x_axis_img{}.png'.format(t + time_step))
        plt.savefig(fname,dpi=200) # save image 
        plt.close() # close figure 

    print(f"photos saved to {folder_dir}")


import cv2
import glob

def create_video(image_folder, output_video, frame_rate=30):
    """
    Create a video from sequential images in a folder.
    
    Parameters:
        image_folder (str): Path to the folder containing images.
        output_video (str): Path for the output video file (e.g., 'output.mp4').
        frame_rate (int): Frames per second for the video.
    """
    # Get all image file paths and sort them
    image_files = sorted(glob.glob(f"{image_folder}/*.png"))  # Adjust file extension if needed
    if not image_files:
        print("No images found in the folder!")
        return

    # Read the first image to get frame dimensions
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape

    # Define video writer with codec, frame rate, and frame size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Write each image to the video
    for img_file in image_files:
        frame = cv2.imread(img_file)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_video}")

# # Example usage
# create_video(image_folder="images", output_video="output_video.mp4", frame_rate=30)

def all_voxel_degs_X_plot(brain_data, masked_xyz, X):
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
    fig.suptitle(f"Plot of All X-Axis Masked Voxel Angles (Degrees) Over All Timestamps")
    plt.show()

    return all_angles