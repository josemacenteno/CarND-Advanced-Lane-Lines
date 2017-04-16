#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python

import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from pipeline import *

#warp all test images
#Define M for the images and Region Of Interest
#load calibration coeff:
coeff_in_path = "./machine_generated_files/calibration_parameters.p"
print("Loading calibration coefficients from here:\n\t" + coeff_in_path)
with open(coeff_in_path, 'rb') as p_in:
    # Pickle the 'data' dictionary using the highest protocol available.
    calibration_parameters = pickle.load(p_in)
    p_mtx = calibration_parameters["mtx"]
    p_dist = calibration_parameters["dist"]




print("Calculating distorsion matrix")
first_image = cv2.imread('./test_images/test1.jpg')  
image_shape = (first_image.shape[1], first_image.shape[0])

Y_TOP = image_shape[1] * 0.64 
Y_BOTTOM = image_shape[1]
TOP_W_HALF = 62
TOP_SHIFT = 4

print(image_shape, Y_BOTTOM, Y_TOP, TOP_W_HALF)

roi_src = np.int32(
    [[(image_shape[0] * 0.5) - TOP_W_HALF + TOP_SHIFT, Y_TOP],
    [ (image_shape[0] * 0.16), Y_BOTTOM],
    [ (image_shape[0] * 0.88), Y_BOTTOM],
    [ (image_shape[0] * 0.5) + TOP_W_HALF + TOP_SHIFT, Y_TOP]])
roi_dst = np.int32(
    [[(image_shape[0] * 0.25), 0],
    [ (image_shape[0] * 0.25), image_shape[1]],
    [ (image_shape[0] * 0.75), image_shape[1]],
    [ (image_shape[0] * 0.75), 0]])


# d) use cv2.getPerspectiveTransform() to get M, the transform matrix
M = cv2.getPerspectiveTransform(np.float32(roi_src), np.float32(roi_dst))

print("Applying undistrotion to test images from ./test_images/...")
for image_name in glob.glob('./test_images/*.jpg'):
    image = cv2.imread(image_name)    
    undistorted = cal_undistort(image, p_mtx, p_dist)
    warped = warp(undistorted, M)
    cv2.polylines(warped,[roi_dst],True,(0,0,255), 10)

    cv2.polylines(undistorted,[roi_src],True,(0,0,255), 10)

    
    small = cv2.resize(undistorted,(256, 144))
    small_w = cv2.resize(warped,(256, 144))

    output_name = "./output_images/warped/original_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small)
    output_name = "./output_images/warped/undistorted_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_w)

    output_name = "./output_images/warped/" + image_name.split('/')[-1]
    cv2.imwrite(output_name, undistorted)