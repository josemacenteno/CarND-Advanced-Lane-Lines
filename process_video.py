#!/home/jcenteno/anaconda3/envs/carnd-term1/bin/python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from pipeline import *

def process_image(image):
    result = pipeline(image)
    return result

#clip1 = VideoFileClip("project_video.mp4")
clip1 = VideoFileClip("short.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

white_output = 'short_output.mp4'

white_clip.write_videofile(white_output, audio=False)