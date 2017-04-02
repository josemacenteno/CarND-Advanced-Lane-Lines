#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#############################################################
# Functions from quizes

def chessboard_draw(img, nx, ny):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    return img


def cal_undistort(img, objpoints, imgpoints):
    """ 
    cal_undistort takes an image, object points, and image points
    performs the camera calibration, image distortion correction and 
    returns the undistorted image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)  
    return undist


def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        deriv_tuple = (1, 0)
    else:
        deriv_tuple = (0, 1)
    sob = cv2.Sobel(gray, cv2.CV_64F, deriv_tuple[0], deriv_tuple[1])
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sob)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def find_image_points(cal_image_file_names, object_points,  display_images = True):
    img_p_list = []
    obj_p_list = []
    for cal_image_filename in cal_images:
        cal_image = cv2.imread(cal_image_filename)
        gray = cv2.cvtColor(cal_image,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        print(ret)
        if ret:
            img_p_list.append(corners)
            obj_p_list.append(object_points)

            # Draw and display the corners
            if display_images:
                cal_image = cv2.drawChessboardCorners(cal_image, (9,6), corners, ret)
                cv2.imshow('img',cal_image)
                cv2.waitKey(500)
        
    # If found, add object points, image points
    return np.array(img_p_list), np.array(obj_p_list)



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
cal_images = glob.glob('./camera_cal/calibration*.jpg')
imgpoints, objpoints = find_image_points(cal_images, objp, False)

for image_name in glob.glob('./test_images/*.jpg'):
    image = cv2.imread(image_name)    
    undistorted = cal_undistort(image, objpoints, imgpoints)

    output_name = "./output_images/" + image_name.split('/')[-1]
    cv2.imwrite(output_name, undistorted)
    #cv2.imshow('img',undistorted)
    cv2.waitKey(500)
