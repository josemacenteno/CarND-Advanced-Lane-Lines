#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python

import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from pipeline import *

# prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...
#                             (0,1,0), (1,1,0), (2,1,0), ...
#                                                      ...., (8,5,0)
# z coordinates are always 0, x and y form a grid
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all calibration images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
print("Reading calibration images from: ./camera_cal/...")
cal_images = glob.glob('./camera_cal/calibration*.jpg')

#Find image points
print("Finding image points")
objpoints, imgpoints = find_image_points(cal_images, objp, False)


#Calculate distortion coefficients
print("Calculating distorsion matrix")
first_image = cv2.imread('./camera_cal/calibration1.jpg')  
image_shape = (first_image.shape[1], first_image.shape[0])
mtx, dist = calibrate_cam(objpoints, imgpoints, image_shape)
calibration_parameters = {"mtx":mtx, "dist":dist}

#Undistort all test images
print("Applying undistrotion to test images from ./test_images/...")
for image_name in glob.glob('./test_images/test*.jpg'):
    image = cv2.imread(image_name)    
    undistorted = cal_undistort(image, mtx, dist)

    small = cv2.resize(image,(256, 144))
    small_u = cv2.resize(undistorted,(256, 144))

    output_name = "./output_images/camera_calibration/original_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small)
    output_name = "./output_images/camera_calibration/undistorted_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_u)

    output_name = "./output_images/camera_calibration/" + image_name.split('/')[-1]
    cv2.imwrite(output_name, undistorted)
    cv2.imshow('img',undistorted)
    cv2.waitKey(500)

#Undistort all chessboard images
print("Applying undistrotion to chessboard images from ./test_images/...")
for image_name in cal_images:
    image = cv2.imread(image_name)    
    undistorted = cal_undistort(image, mtx, dist)
    small = cv2.resize(image,(256, 144))
    small_u = cv2.resize(undistorted,(256, 144))

    output_name = "./output_images/camera_calibration/original_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small)
    output_name = "./output_images/camera_calibration/undistorted_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_u)
    cv2.imshow('img',undistorted)
    cv2.waitKey(500)

 