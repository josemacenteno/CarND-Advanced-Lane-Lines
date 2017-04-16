##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[distort2]: ./output_images/camera_calibration/original_calibration2.jpg "Disistorted chessboard"
[undistort2]: ./output_images/camera_calibration/undistorted_calibration2.jpg "Undistorted chessboard"
[distort3]: ./output_images/camera_calibration/original_calibration3.jpg "Disistorted chessboard"
[undistort3]: ./output_images/camera_calibration/undistorted_calibration3.jpg "Undistorted chessboard"
[distort5]: ./output_images/camera_calibration/original_calibration5.jpg "Disistorted chessboard"
[undistort5]: ./output_images/camera_calibration/undistorted_calibration5.jpg "Undistorted chessboard"
[distort6]: ./output_images/camera_calibration/original_calibration6.jpg "Disistorted chessboard"
[undistort6]: ./output_images/camera_calibration/undistorted_calibration6.jpg "Undistorted chessboard"
[distort7]: ./output_images/camera_calibration/original_calibration7.jpg "Disistorted chessboard"
[undistort7]: ./output_images/camera_calibration/undistorted_calibration7.jpg "Undistorted chessboard"
[distort8]: ./output_images/camera_calibration/original_calibration8.jpg "Disistorted chessboard"
[undistort8]: ./output_images/camera_calibration/undistorted_calibration8.jpg "Undistorted chessboard"


[test_d1]: ./output_images/camera_calibration/original_test1.jpg "Disistorted test1"
[test_u1]: ./output_images/camera_calibration/undistorted_test1.jpg "Undistorted test1"
[test_d2]: ./output_images/camera_calibration/original_test2.jpg "Disistorted test2"
[test_u2]: ./output_images/camera_calibration/undistorted_test2.jpg "Undistorted test2"
[test_d3]: ./output_images/camera_calibration/original_test3.jpg "Disistorted test3"
[test_u3]: ./output_images/camera_calibration/undistorted_test3.jpg "Undistorted test3"
[test_d4]: ./output_images/camera_calibration/original_test4.jpg "Disistorted test4"
[test_u4]: ./output_images/camera_calibration/undistorted_test4.jpg "Undistorted test4"
[test_d5]: ./output_images/camera_calibration/original_test5.jpg "Disistorted test5"
[test_u5]: ./output_images/camera_calibration/undistorted_test5.jpg "Undistorted test5"
[test_d6]: ./output_images/camera_calibration/original_test6.jpg "Disistorted test6"
[test_u6]: ./output_images/camera_calibration/undistorted_test6.jpg "Undistorted test6"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the script  code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][distort2]        ![alt text][undistort2]

![alt text][distort3]        ![alt text][undistort3]

![alt text][distort5]        ![alt text][undistort5]

![alt text][distort6]        ![alt text][undistort6]

![alt text][distort7]        ![alt text][undistort7]

![alt text][distort8]        ![alt text][undistort8]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will show all the test images before and after applying distortion correction


![alt text][test_d2]        ![alt text][test_u2]

![alt text][test_d3]        ![alt text][test_u3]

![alt text][test_d5]        ![alt text][test_u5]

![alt text][test_d6]        ![alt text][test_u6]

![alt text][test_d7]        ![alt text][test_u7]

![alt text][test_d8]        ![alt text][test_u8]


![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

