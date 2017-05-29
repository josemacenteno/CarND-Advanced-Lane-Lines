#!/home/jcenteno/anaconda3/envs/carnd-term1/bin/python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from pipeline import *



frame_counter = 0
def process_image(image):
    global frame_counter
    result = pipeline(image)

    output_name = "./video_clip/video" + str(frame_counter) + ".jpg"
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    cv2.imwrite(output_name, bgr_image)

    # cv2.imshow("img", result)
    # cv2.waitKey(10)
    frame_counter += 1
    return result


#clip1 = VideoFileClip("project_video.mp4")
clip1 = VideoFileClip("project_video.mp4")
# clip1 = VideoFileClip("project_video.mp4").subclip(41,43)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

# white_output = 'subclip_project_video_output.mp4'
white_output = 'project_video_output.mp4'

white_clip.write_videofile(white_output, audio=False)