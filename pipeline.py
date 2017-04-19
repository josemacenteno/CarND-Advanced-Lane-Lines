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


def chessboard_draw(img, nx, ny, gray=False):
    if gray:
        gray = img
    else:
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


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(10, 200), gray = False):
    # Calculate directional gradient
    # Apply threshold
    # 1) Convert to grayscale
    if gray:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        deriv_tuple = (1, 0)
    else:
        deriv_tuple = (0, 1)
    sob = cv2.Sobel(gray, cv2.CV_64F, deriv_tuple[0], deriv_tuple[1], ksize = sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sob)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your grad_binary image
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(40, 120), gray=False):
    # Calculate gradient magnitud
    # Apply threshold
    # 1) Convert to grayscale
    if gray:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobel = np.sqrt(sobx**2 + soby**2)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0.8, 1.25), gray=False):
    # Calculate gradient direction
    # Apply threshold
    # 1) Convert to grayscale
    if gray:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.sqrt(sobx**2)
    abs_sobely = np.sqrt(soby**2)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direct_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where mag thresholds are met
    dir_binary = np.zeros_like(direct_sobel)
    dir_binary[(direct_sobel >= thresh[0]) & (direct_sobel <= thresh[1])] = 1

    # 6) Return this mask as your dir_binary image
    return dir_binary


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_filter(img, thresh=(60, 255)):
    # 1) Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    img_s = img_hls[:,:,2]
    
    # 3) Return a binary image of threshold result
    hls_binary = np.zeros_like(img_s)
    lt = img_s > thresh[0]
    ht = img_s <= thresh[1]
    in_t = lt & ht
    hls_binary[in_t] = 1
    return hls_binary

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def bgr_to_s(img):
    # 1) Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    img_s = img_hls[:,:,2]
    eq_s = cv2.equalizeHist(img_s).reshape(img_s.shape)
    return eq_s


def pipeline(img):
    ksize = 3
    hls_eq = bgr_to_s(img)
    gradx = abs_sobel_thresh(hls_eq, orient='x', sobel_kernel=ksize, thresh=(40, 180), gray=True)
    grady = abs_sobel_thresh(hls_eq, orient='y', sobel_kernel=ksize, thresh=(60, 200), gray=True)
    mag_binary = mag_thresh(hls_eq, sobel_kernel=ksize, mag_thresh=(60, 120), gray=True)
    dir_binary = dir_threshold(hls_eq, sobel_kernel=ksize, thresh=(0.7, 1.5), gray=True)


    combined = np.zeros_like(dir_binary)
    combined[((mag_binary == 1) & (dir_binary == 1)) | ((gradx == 1) & (grady == 1))] = np.uint8((255))
    three_chan = np.dstack((np.zeros_like(combined), np.zeros_like(combined), combined))
    α=0.8
    β=1.
    λ=0
    print(img.shape, three_chan.shape)
    superposed = cv2.addWeighted(np.uint8(img), α, np.uint8(three_chan), β, λ)
    return superposed
