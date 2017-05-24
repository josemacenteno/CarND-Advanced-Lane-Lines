#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

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

print(image_shape, Y_BOTTOM, Y_TOP, TOP_W_HALF)
print("roi_src:", roi_src, "roi_dst:", roi_dst)
# d) use cv2.getPerspectiveTransform() to get M, the transform matrix
M = cv2.getPerspectiveTransform(np.float32(roi_src), np.float32(roi_dst))
M_inv = cv2.getPerspectiveTransform(np.float32(roi_dst), np.float32(roi_src))

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
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    return img

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(10, 200), gray = False):
    # Calculate directional gradient
    # Apply threshold
    # 1) Convert to grayscale
    if gray:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
def hls_select(img, thresh=(0, 255), index=2):
    # 1) Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    img_s = img_hls[:,:,index]
    
    # 3) Return a binary image of threshold result
    img_bin = np.zeros_like(img_s)
    lt = img_s > thresh[0]
    ht = img_s <= thresh[1]
    in_t = lt & ht
    img_bin[in_t] = 1
    return img_bin

def RGB_to_h_l_s(img, index=2):
    # 1) Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2RGB)
    # 2) Apply a threshold to the S channel
    img_s = img_hls[:,:,index]
    eq_s = cv2.equalizeHist(img_s).reshape(img_s.shape)
    return eq_s

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def RGB_to_r_g_b(img, index=2):
    # 2) Apply a threshold to the S channel
    img_1ch = img[:,:,index]
    eq_img = cv2.equalizeHist(img_1ch).reshape(img_1ch.shape)
    return eq_img

def min_max_scale(instance):
    x_min = 0
    x_max = 255
    a, b = 0.0, 1.0
    conversion_factor = (b - a)/(x_max - x_min)    
    return np.add(a,np.multiply(np.subtract(instance,x_min), conversion_factor))

def pre_process(x_in, y_in):
    if use_gray_images:
        gray_instance = cv2.cvtColor(instance, cv2.COLOR_RGB2GRAY).reshape(gray_image_shape)
        instance = gray_instance
    eq_instance = cv2.equalizeHist(instance).reshape(instance.shape)
    return min_max_scale(eq_instance)

def merge(img, overlay):
    α=0.8
    β=1.
    λ=0
    superposed = cv2.addWeighted(np.uint8(img), α, np.uint8(overlay), β, λ)
    return superposed

def draw_lane(binary_warped):
    left_fit, right_fit = find_lane(binary_warped)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp(color_warp, M_inv) 

    return newwarp


def find_lane(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def pipeline(img,
             return_bin_threshold = False,
             hthresh = (16, 23),
             lthresh = (209, 250),
             sthresh = (140, 250),
             xthresh = (20, 60),
             ythresh = (30, 120),
             mthresh = (50, 150),
             tthresh = (0.60, 1.4)
             ):
    ksize = 3
    
    undistorted = cal_undistort(img, p_mtx, p_dist)

    if return_bin_threshold:
        warped = undistorted
    else:
        warped = warp(undistorted, M)

    h_bin = hls_select(warped, thresh=hthresh, index=0)
    l_bin = hls_select(warped, thresh=lthresh, index=1)
    s_bin = hls_select(warped, thresh=sthresh, index=2)
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=xthresh, gray=True)
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=ythresh, gray=True)
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=mthresh, gray=True)
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=tthresh, gray=True)
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1) ) |                  \
             ((mag_binary == 1) & (dir_binary== 1) ) |         \
             ((s_bin == 1) & ((h_bin == 1) | (l_bin== 1))) ]         \
                                                          = np.uint8(1)
    gray_combined = 255 * combined
    three_chan = np.dstack((gray_combined, gray_combined, gray_combined))

    if return_bin_threshold:
        return  gray_combined
        

    lane_drawing = draw_lane(combined)

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, lane_drawing, 0.3, 0)
    

    return result

