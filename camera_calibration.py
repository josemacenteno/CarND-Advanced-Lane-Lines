#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python

import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

#Code based on class quiz
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

def calibrate_cam(objpoints, imgpoints, img_shape):
    """ 
    cal_undistort takes an image, object points, and image points
    performs the camera calibration, image distortion correction and 
    returns the undistorted image
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    return mtx, dist

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

coeff_out_path = "./machine_generated_files/calibration_parameters.p"
print("saving calibration coefficients here:\n\t" + coeff_out_path)
with open(coeff_out_path, 'wb') as p_out:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(calibration_parameters, p_out)

print("Done")