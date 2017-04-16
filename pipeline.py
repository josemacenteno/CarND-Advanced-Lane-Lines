#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python

import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

#load calibration coeff:
coeff_in_path = "./machine_generated_files/calibration_parameters.p"
print("Loading calibration coefficients from here:\n\t" + coeff_in_path)
with open(coeff_in_path, 'rb') as p_in:
    # Pickle the 'data' dictionary using the highest protocol available.
    calibration_parameters = pickle.load(p_in)
    mtx = calibration_parameters["mtx"]
    dist = calibration_parameters["dist"]


#print(mtx)
#print(dist)
