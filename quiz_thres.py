#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pipeline import *
    


image_name = "./test_images/quiz_thres_signs_vehicles_xygrad.png"
image = cv2.imread(image_name)    

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions

hls_binary = hls_filter(image, thresh=(60, 255))

hls_eq = bgr_to_s(image)
gradx = abs_sobel_thresh(hls_eq, orient='x', sobel_kernel=ksize, thresh=(40, 180), gray=True)
grady = abs_sobel_thresh(hls_eq, orient='y', sobel_kernel=ksize, thresh=(60, 200), gray=True)
mag_binary = mag_thresh(hls_eq, sobel_kernel=ksize, mag_thresh=(60, 120), gray=True)
dir_binary = dir_threshold(hls_eq, sobel_kernel=ksize, thresh=(0.7, 1.5), gray=True)


combined = np.zeros_like(dir_binary)
combined[((mag_binary == 1) & (dir_binary == 1)) | ((gradx == 1) & (grady == 1))] = 1

comb_xy = np.zeros_like(dir_binary)
comb_xy[((gradx == 1) & (grady == 1))] = 1

comb_ht = np.zeros_like(dir_binary)
comb_ht[((mag_binary == 1) & (dir_binary == 1))] = 1

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.imshow(combined, cmap='gray')
plt.show()