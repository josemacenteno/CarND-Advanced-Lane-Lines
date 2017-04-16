#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip




def cal_undistort(img, mtx, dist):
    """ 
    cal_undistort takes an image, object points, and image points
    performs the camera calibration, image distortion correction and 
    returns the undistorted image
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)  
    return undist

def warp(img, M):
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR) 
    return warped


def chessboard_draw(img, nx, ny):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    return img


def calibrate_cam(objpoints, imgpoints, img_shape):
    """ 
    cal_undistort takes an image, object points, and image points
    performs the camera calibration, image distortion correction and 
    returns the undistorted image
    """

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    return mtx, dist



def find_image_points(cal_image_file_names, object_points,  display_images = True):
    img_p_list = []
    obj_p_list = []
    for cal_image_filename in cal_image_file_names:
        cal_image = cv2.imread(cal_image_filename)
        gray = cv2.cvtColor(cal_image,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        if not ret:
            print("Couldn't find all points in", cal_image_filename)
        else:
            img_p_list.append(corners)
            obj_p_list.append(object_points)

            # Draw and display the corners
            if display_images:
                cal_image = cv2.drawChessboardCorners(cal_image, (9,6), corners, ret)
                cv2.imshow('img',cal_image)
                cv2.waitKey(500)
        
    # If found, add object points, image points
    return np.array(obj_p_list), np.array(img_p_list)


