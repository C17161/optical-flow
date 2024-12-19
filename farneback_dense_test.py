# MODIFICATION: instead of drawing on the lines continuously, take one frame to the next and draw arrows
# in the general direction that the vector is going

# FARNEBACK AND DENSE PYRAMID LK AND DENSE OPTICAL FLOW
# from one frame to the other (t1 to t2), get vector motion estimates of direction travelled

# LOAD DATA (DROPBOX), see if you can plot time series, etc
# use ventricle mask to cut coordinates to show xyz planes (pull out frames we will use -- centered on ventricles)
# from mask img, choose what coord system to slice data on -- 1.91 KB

# # DENSE OPTICAL FLOW GITHUB IMPLEMENTATAION
# import numpy as np
# import cv2 as cv 
# # the actual video mp4 file directory path
# cap = cv.VideoCapture('/Users/catherinetu/Documents/Opticalflow/slow_traffic_small.mp4')
# num = 54
# num2 = 63
# for i in range(num):
#     ret, frame1 = cap.read()
# # converts frame1 to grayscale
# prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
# # returns array of zeros with same shape & data type
# hsv = np.zeros_like(frame1)
# print(len(hsv))
# hsv[..., 1] = 255
# while(1):
#     ret, frame2 = cap.read()
#     if not ret:
#         print('No frames grabbed!')
#         # break
#     next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
#     flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang*180/np.pi/2
#     hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
#     bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#     cv.imshow('frame2', bgr)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         # break
#         pass
#     elif k == ord('s'):
#         cv.imwrite('opticalfb.png', frame2)
#         cv.imwrite('opticalhsv.png', bgr)
#     prvs = next
# cv.destroyAllWindows()



# DENSE OPTICAL FLOW with vector fields b/t 2 frames
# vector field on whole video
# import numpy as np
# import cv2 as cv

# # Initialize video capture
# cap = cv.VideoCapture('/Users/catherinetu/Documents/Opticalflow/slow_traffic_small.mp4')

# # Read the initial frame
# ret, frame1 = cap.read()
# if not ret:
#     print('Error: Could not read the first frame.')
#     cap.release()
#     cv.destroyAllWindows()
#     exit()

# # Convert to grayscale
# prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# # Define parameters
# step = 15  # Set the step size for subsampling flow vectors
# color = (0, 255, 0)  # Set arrow color (e.g., green)

# while True:
#     # Read the next frame
#     ret, frame2 = cap.read()
#     if not ret:
#         print('No more frames to read.')
#         break

#     # Convert to grayscale
#     next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

#     # Calculate dense optical flow
#     flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # Draw arrows on the frame
#     for y in range(0, flow.shape[0], step):
#         for x in range(0, flow.shape[1], step):
#             fx, fy = flow[y, x]  # Flow vector at (x, y)
#             end_point = (int(x + fx), int(y + fy))  # End point of the arrow
#             cv.arrowedLine(frame2, (x, y), end_point, color, 1, tipLength=0.3)

#     # Display the result
#     cv.imshow('Optical Flow - Arrows', frame2)

#     # Break on 'Esc' key
#     if cv.waitKey(30) & 0xff == 27:
#         break

#     # Update previous frame
#     prvs = next

# # Release and close
# cap.release()
# cv.destroyAllWindows()




import numpy as np
import cv2 as cv

# Initialize video capture
cap = cv.VideoCapture('/Users/catherinetu/Documents/Opticalflow/slow_traffic_small.mp4')

# User-defined frames to compare
# start_frame = int(input("Enter the starting frame number: "))
# end_frame = int(input("Enter the ending frame number: "))
start_frame = 3
end_frame = 5

# Set the video to the starting frame
# cv.CAP_PROP_POS_FRAMES signifies we want to jump to a specific frame (start_frame)
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
# ret = bool indicating if frame was successfully read & frame1 = image itself
ret, frame1 = cap.read()
if not ret:
    print(f"Error: Could not read frame {start_frame}.")
    cap.release()
    cv.destroyAllWindows()
    exit()

# Convert to grayscale
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# Move to the ending frame and capture it
cap.set(cv.CAP_PROP_POS_FRAMES, end_frame)
ret, frame2 = cap.read()
if not ret:
    print(f"Error: Could not read frame {end_frame}.")
    cap.release()
    cv.destroyAllWindows()
    exit()

# Convert to grayscale
next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

# Calculate dense optical flow between the two frames
flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# visualize
# plt.imshow(flow[:,:,1]);plt.show()


# X, Y = np.meshgrid(np.arange(next.shape[1]),np.arange(next.shape[0]))

# fig1, ax1 = plt.subplots()

# Q = ax1.quiver(X[:10,:10], Y[:10,:10], flow[:10,:10,0], flow[:10,:10,1], units='width')
# plt.show()

type(flow)

# Draw arrows on the ending frame
step = 15  #spacing bt arrows in flow visual
color = (0, 255, 0)  # arrow color BGR
scalar = 10
# iterating over flow field with step in both x and y
for y in range(0, flow.shape[0], step):
    for x in range(0, flow.shape[1], step):
        fx, fy = flow[y, x]  # extracts optical flow vector at (x, y)
        end_point = (int(x + fx*scalar), int(y + fy*scalar))  # calc end point of the arrow using flow vector
        # draw arrows from x, y to end_point
        cv.arrowedLine(frame2, (x, y), end_point, color, 1, tipLength=.1)

# Display the result with arrows drawn
cv.imshow('Optical Flow - Arrows', frame2)
cv.waitKey(0)  # Wait until a key is pressed to close

# Release and close
cap.release()
cv.destroyAllWindows()




# def dense_optical_flow(method, video_path, params=[], to_gray=False):
#     # Read the video and first frame
#     cap = cv.VideoCapture(video_path)
#     ret, old_frame = cap.read()
 
#     # crate HSV & make Value a constant
#     hsv = np.zeros_like(old_frame)
#     hsv[..., 1] = 255
 
#     # Preprocessing for exact method
#     if to_gray:
#         old_frame = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
#     while True:
#         # Read the next frame
#         ret, new_frame = cap.read()
#         frame_copy = new_frame
#         if not ret:
#             break
    
#         # Preprocessing for exact method
#         if to_gray:
#             new_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
    
#         # Calculate Optical Flow
#         flow = method(old_frame, new_frame, None, *params)
    
#         # Encoding: convert the algorithm's output into Polar coordinates
#         mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
#         # Use Hue and Value to encode the Optical Flow
#         hsv[..., 0] = ang * 180 / np.pi / 2
#         hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    
#         # Convert HSV image into BGR for demo
#         bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#         cv.imshow("frame", frame_copy)
#         cv.imshow("optical flow", bgr)
#         k = cv.waitKey(25) & 0xFF
#         if k == 27:
#             break
    
#         # Update the previous frame
#         old_frame = new_frame

# dense_optical_flow(cv.optflow.calcOpticalFlowSparseToDense, '/Users/catherinetu/Documents/Opticalflow/slow_traffic_small.mp4',
#                    [8, 128, 0.05, True, 500, 1.5])






# # working cv arrows but it kills my computer soooo maybe not
# import numpy as np
# import cv2 as cv

# # Initialize video capture
# cap = cv.VideoCapture('/Users/catherinetu/Documents/Opticalflow/slow_traffic_small.mp4')
# fig, ax = plt.subplots()

# start_frame = 0
# end_frame = 50
# # initialize total_flow that will sum all flow frames that is a certain shape dims
# shape = (360, 640, 2)
# total_flow = [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
# # used for image
# last_frame = [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
# all_flow = []
# for frame in range(start_frame, end_frame + 1):
#     # Set the video to the starting frame
#     # cv.CAP_PROP_POS_FRAMES signifies we want to jump to a specific frame (start_frame)
#     cap.set(cv.CAP_PROP_POS_FRAMES, frame)
#     # ret = bool indicating if frame was successfully read & frame1 = image itself
#     ret, current_frame = cap.read()
#     if not ret:
#         print(f"Error: Could not read frame {current_frame}.")
#         cap.release()
#         cv.destroyAllWindows()
#         exit()

#     # Convert to grayscale
#     current_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

#     cap.set(cv.CAP_PROP_POS_FRAMES, frame + 1)
#     # ret = bool indicating if frame was successfully read & frame1 = image itself
#     ret, next_frame = cap.read()
#     if not ret:
#         print(f"Error: Could not read frame {next_frame}.")
#         cap.release()
#         cv.destroyAllWindows()
#         exit()

#     next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

#     # Calculate dense optical flow between the two frames
#     frame_flow = cv.calcOpticalFlowFarneback(current_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     all_flow.append(frame_flow)
#     last_frame = next_frame

# all_flow = np.asarray(all_flow)
# # calculate mean total_flow
# mean_flow = np.mean(all_flow,axis=0)

# # Draw arrows on the ending frame
# step = 15  #spacing bt arrows in flow visual
# color = (0, 255, 0)  # arrow color BGR
# scalar = 100
# # iterating over flow field with step in both x and y
# for y in range(0, mean_flow.shape[0], step):
#     for x in range(0, mean_flow.shape[1], step):
#         fx, fy = mean_flow[y,x]  # extracts optical flow vector at (x, y)
#         end_point = (int(x + fx * scalar), int(y + fy * scalar))  # calc end point of the arrow using flow vector
#         # draw arrows from x, y to end_point
#         cv.arrowedLine(last_frame, (x, y), end_point, color = (255, 0, 0), thickness=1, tipLength = 1)
# plt.show()

# # Display the result with arrows drawn
# cv.imshow('Optical Flow Average', last_frame)
# cv.waitKey(0)
# # Release and close
# cap.release()
# cv.destroyAllWindows()



# 11/8/24 NEXT STEPS:
# 1: plot the average optical flow using arrows over x number of frames, given a starting and end frame.
# 2: try to plot optical flow onto brain data

import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.cm as cm # for color

# Initialize video capture
# cap = cv.VideoCapture('/Users/catherinetu/Documents/Opticalflow/slow_traffic_small.mp4')
cap = cv.VideoCapture('/Users/catherinetu/Documents/Opticalflow/slow_traffic_small.mp4')
fig, ax = plt.subplots()

start_frame = 50
end_frame = 100
# initialize total_flow that will sum all flow frames that is a certain shape dims
shape = (360, 640, 2)
total_flow = [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
# used for image
last_frame = [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
all_flow = []
for frame in range(start_frame, end_frame + 1):
    # Set the video to the starting frame
    # cv.CAP_PROP_POS_FRAMES signifies we want to jump to a specific frame (start_frame)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame)
    # ret = bool indicating if frame was successfully read & frame1 = image itself
    ret, current_frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {current_frame}.")
        cap.release()
        cv.destroyAllWindows()
        exit()

    # Convert to grayscale
    current_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

    cap.set(cv.CAP_PROP_POS_FRAMES, frame + 1)
    # ret = bool indicating if frame was successfully read & frame1 = image itself
    ret, next_frame = cap.read()
    ax.imshow(next_frame)
    if not ret:
        print(f"Error: Could not read frame {next_frame}.")
        cap.release()
        cv.destroyAllWindows()
        exit()

    next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

    # Calculate dense optical flow between the two frames
    frame_flow = cv.calcOpticalFlowFarneback(current_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    all_flow.append(frame_flow)
    last_frame = next_frame

all_flow = np.asarray(all_flow)
# calculate mean total_flow
mean_flow = np.mean(all_flow,axis=0)

# close opencv2 stuff

#plot with matplotlib
# plot.arrow(x, y, dx, dy)

# Draw arrows on the ending frame
step = 15  #spacing bt arrows in flow visual
color = (0, 255, 0)  # arrow color BGR
scalar = 30
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
        arrow = patches.Arrow(x, y, dx=end_point[0], dy=end_point[1], width=5, color=color_ang)
        ax.add_patch(arrow)
        # cv.arrowedLine(last_frame, (x, y), end_point, color = (255, 0, 0), thickness=1, tipLength = 1)
# fig.set_title("Optical Flow Vectors")
plt.show()




# explore more outputs of flow across time -- what do we do with computed data over time
# look at data more and what it looks like
    # converting to polar coords
    # look for single pixel: how is theta distributed over time (is it uniformly dist??)
    # is something uniformly distributed??
    # kl divergence test -- on right track
# pick x or y slice on brain data (since z slice is pretty stagnant)
    # use mask
    # slice & compute flow & plot average flow over time
# simulation -- described in email too
    # generate data where we have known underlying parameters and see how well we can get them
# presentation