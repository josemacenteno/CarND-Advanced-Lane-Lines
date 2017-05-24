#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python

import pickle
import numpy as np
import cv2
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from pipeline import *

#warp all test images

# Undistort all chessboard images
print("Applying undistrotion to chessboard images from ./test_images/...")
for image_name in glob.glob('./camera_cal/calibration*.jpg'):
    image = cv2.imread(image_name)    
    undistorted = cal_undistort(image, p_mtx, p_dist)
    small = cv2.resize(image,(256, 144))
    small_u = cv2.resize(undistorted,(256, 144))

    output_name = "./output_images/camera_calibration/original_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small)
    output_name = "./output_images/camera_calibration/undistorted_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_u)
    cv2.imshow('img',undistorted)
    cv2.waitKey(50)

#Undistort all test images
print("Applying undistrotion to test images from ./test_images/...")
for image_name in glob.glob('./test_images/test*.jpg'):
    image = cv2.imread(image_name)    
    undistorted = cal_undistort(image, p_mtx, p_dist)

    small = cv2.resize(image,(256, 144))
    small_u = cv2.resize(undistorted,(256, 144))

    output_name = "./output_images/camera_calibration/original_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small)
    output_name = "./output_images/camera_calibration/undistorted_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_u)

    output_name = "./output_images/camera_calibration/" + image_name.split('/')[-1]
    cv2.imwrite(output_name, undistorted)
    cv2.imshow('img',undistorted)
    cv2.waitKey(50)


print("Applying warp to test images from ./test_images/...")
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
    output_name = "./output_images/warped/warp_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_w)

    output_name = "./output_images/warped/" + image_name.split('/')[-1]
    cv2.imwrite(output_name, warped)
    cv2.imshow('img', warped)
    cv2.waitKey(50)

#Aplying binary thresholding to all test images
print("Applying pipeline to test images from ./test_images/...")
for image_name in glob.glob('./test_images/test*.jpg'):
    image = cv2.imread(image_name)
    bin_thres = pipeline(image, return_bin_threshold = True,
             hthresh = (16, 23),
             lthresh = (209, 250),
             sthresh = (140, 250),
             xthresh = (20, 60),
             ythresh = (30, 120),
             mthresh = (50, 150),
             tthresh = (0.60, 1.4))

    small = cv2.resize(image,(256, 144))
    small_p = cv2.resize(bin_thres,(256, 144))

    output_name = "./output_images/bin_thres/original_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small)
    output_name = "./output_images/bin_thres/bin_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_p)

    output_name = "./output_images/bin_thres/" + image_name.split('/')[-1]
    cv2.imwrite(output_name, bin_thres)

    cv2.imshow('img',bin_thres)
    cv2.waitKey(50)


#Aplying pipeline to all test images
print("Applying pipeline to test images from ./test_images/...")
for image_name in glob.glob('./test_images/test*.jpg'):
    image = cv2.imread(image_name)    
    print(image_name)
    pipelined = pipeline(image)
    small = cv2.resize(image,(256, 144))
    small_p = cv2.resize(pipelined,(256, 144))

    output_name = "./output_images/pipelined/original_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small)
    output_name = "./output_images/pipelined/piped_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_p)

    output_name = "./output_images/pipelined/" + image_name.split('/')[-1]
    cv2.imwrite(output_name, pipelined)

    cv2.imshow('img',pipelined)
    cv2.waitKey(500)

