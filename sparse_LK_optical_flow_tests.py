# OPTICAL FLOW LUCAS - KANADE
# try given examples
# next step(s): try optical flow on real data and see what happens
# simulate cases (which ones can we pick up flow? which ones can we not see flows?)
# create your own visualization? if ventricle at edge, is it more accurate?

# import argparse
# CODE FOR RUNNING FROM TERMINAL
# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              # The example file can be downloaded from: \
                                              # https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
#parser.add_argument('image', type=str, help='path to image file')
#args = parser.parse_args()

# MODIFICATION: instead of drawing on the lines continuously, take one frame to the next and draw arrows
# in the general direction that the vector is going

# FARNEBACK AND DENSE PYRAMID LK AND DENSE OPTICAL FLOW
# from one frame to the other (t1 to t2), get vector motion estimates of direction travelled

# LOAD DATA (DROPBOX), see if you can plot time series, etc
# use ventricle mask to cut coordinates to show xyz planes (pull out frames we will use -- centered on ventricles)
# from mask img, choose what coord system to slice data on -- 1.91 KB

import numpy as np
import cv2 as cv 
image_path = '/Users/catherinetu/Documents/Opticalflow/slow_traffic_small.mp4'
cap = cv.VideoCapture(image_path)

# params for ShiTomasi corner detection
# ShiTomasi corner detection: finds N strongest corners in the image
    # maxCorners: max corners we want to detect. if more are found, the 100 strongest get returned
    # qualityLevel: min quality level of corner accepted
    # minDistance: min Euclidean distance between corners
    # blockSize: size of an average block 'for computing a derivative covariation matrix over each pixel neighborhood'
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
    # winSize = size of search window at each pyramid level
        # larger window sizes can handle larger motion but will be less precise
    # maxLevel = no more than maxLevel of pyramids will be passed
        # higher levels handle more motion but increase computational cost
    # criteria: specifying the termination criteria of the iterative search algorithm
        # TERM_CRITERIA_EPS terminates when error < epsilon (0.03)
        # TERM_CRITERIA_COUNT terminates after a certain number of iterations (10)
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
# ranging from 0 to 255, generate 100 arrays each with 3 elements inside
color = np.random.randint(0, 255, (100, 3))
# Take FIRST frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY) # converting colored frame to greyscale
# p0 contains key coordinates of feature points
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image -- filled with zeros for drawing purposes
# algorithm will ignore these areas when detecting features
mask = np.zeros_like(old_frame)
# lucas - kanade optical flow implementation
while(1):
    ret, frame = cap.read()
    # if no more frames available
    if not ret:
        print('No frames grabbed!')
        break
    # converts image to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
        # old_gray = previous frame
        # frame_gray = current frame
        # p0 = previously tracked points
        # None = indicates we haven't generated next points yet
        # lk_params = parameters for lucas kanade algorithm
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # p1 = new positions of tracked points
        # st = status array indicating success / failures of each point
        # err = error measure for each point
    # Select good points that were successfully tracked
    if p1 is not None:
        good_new = p1[st == 1] # st == 1 selects only points that were successfully tracked
        good_old = p0[st == 1]
    # draw the tracks connecting current tracked positions to old
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel() # ravel() changes multi-D array to contigious flat array
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    # displays the frame with drawn tracks and waits for a key press. Pressing 'Esc' breaks the loop.
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cv.destroyAllWindows()





# import numpy as np
# import cv2 as cv 
# # print(np.__version__)
# # print(cv.__version__)
# image_path = '/Users/catherinetu/Documents/Opticalflow/slow_traffic_small.mp4'
# cap = cv.VideoCapture(image_path)

# if not cap.isOpened():
#     print('cannot open camera -- exiting')
#     exit()

# # params for ShiTomasi corner detection
# # ShiTomasi corner detection: finds N strongest corners in the image
#     # maxCorners: max corners we want to detect. if more are found, the 100 strongest get returned
#     # qualityLevel: min quality level of corner accepted
#     # minDistance: min Euclidean distance between corners
#     # blockSize: size of an average block 'for computing a derivative covariation matrix over each pixel neighborhood'
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# # Parameters for lucas kanade optical flow
#     # winSize = size of search window at each pyramid level
#         # larger window sizes can handle larger motion but will be less precise
#     # maxLevel = no more than maxLevel of pyramids will be passed
#         # higher levels handle more motion but increase computational cost
#     # criteria: specifying the termination criteria of the iterative search algorithm
#         # TERM_CRITERIA_EPS terminates when error < epsilon (0.03)
#         # TERM_CRITERIA_COUNT terminates after a certain number of iterations (10)
# lk_params = dict( winSize  = (15, 15),
#                   maxLevel = 2,
#                   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# # Create some random colors
# # ranging from 0 to 255, generate 100 arrays each with 3 elements inside
# color = np.random.randint(0, 255, (100, 3)).astype(int)
# # Take FIRST frame and find corners in it
# ret, old_frame = cap.read()
# mask = np.zeros_like(old_frame)
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY) # converting colored frame to greyscale
# # p0 contains key coordinates of feature points
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# # Create a mask image -- filled with zeros for drawing purposes
# # algorithm will ignore these areas when detecting features
# mask = np.zeros_like(old_frame)

# # lucas - kanade optical flow implementation
# while cap.isOpened():
#     ret, frame = cap.read()
#     # frame = frame.tolist()
#     # if no more frames available
#     if not ret:
#         print('No frames grabbed!')
#         break
#     # converts image to grayscale
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # calculate optical flow
#         # old_gray = previous frame
#         # frame_gray = current frame
#         # p0 = previously tracked points
#         # None = indicates we haven't generated next points yet
#         # lk_params = parameters for lucas kanade algorithm
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#         # p1 = new positions of tracked points
#         # st = status array indicating success / failures of each point
#         # err = error measure for each point
#     # Select good points that were successfully tracked
#     if p1 is not None:
#         good_new = p1[st == 1] # st == 1 selects only points that were successfully tracked
#         good_old = p0[st == 1]
#     # draw the tracks connecting current tracked positions to old
#     step_interval = 5 # interval between in which to draw the arrows
#     for i in range(0, frame.shape[0], step_interval):
#         for j in range(0, frame.shape[1], step_interval):
#             x, y = j, i
#             u, v, = 0, 0
#             # find nearest good pts
#             min_dist = float('inf')
#             for pt in good_new:
#                 dist = ((x - pt[0])**2 + (y - pt[1])**2)**0.5 # pythagorean theorem
#                 if dist < min_dist:
#                     min_dist = dist
#                     u = pt[0] - x
#                     v = pt[1] - y

#             # Draw arrow
#             end_point = (int(x + u * 10), int(y + v * 10))  # Scale factor of 10
#             # print(color)
#             # print(tuple(color[i%len(color)]))
#             color_tup = tuple(np.random.randint(0, 255, 3).astype(int))
#             cv.arrowedLine(frame, (x, y), end_point, (33, 33, 33), 2);

#             old_gray = frame_gray.copy()
#             p0 = good_new.reshape(-1, 1, 2)
# cap.release()
# cv.imshow('frame', frame)
# cv.destroyAllWindows()
