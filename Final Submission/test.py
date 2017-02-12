# Import all the necessary files
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import all the files with the functions I used
from cameracal import *
from colandgrad import *
from persptrans import *
from slidwindow import *
from draw import *

# Import the file with the class definition
from line import Line

# Import movie editor files
from moviepy.editor import VideoFileClip

def pipelineVideo(image):
    ''' Description: This function applies various stages of the pipeline on the
                     image(which is a single frame captured from the video)

        Inputs: image - Image which is processed

        Outputs: result - Final image with lane drawn and curvature and position
                          offset added to the image
    '''
    # The following two variables are globals(for tracking)
    global line_tracking
    global averaging_threshold

    # Get the image to test
    test_image = image

    # Undistort the image
    undistorted_image = undistortImage(test_image, objpoints, imgpoints, False)

    # Apply color and gradient thresholding
    thresholded_image = colorAndGradientThreshold(img=undistorted_image, display_images=False)

    # Get the perspective transform of the image
    warped_image, Minv = perspectiveTransform(test_image, thresholded_image, False)
    warped = np.copy(warped_image)

    # Apply sliding window
    plot_y, left_fit_x, right_fit_x, line_tracking = slidingWindow(warped, line_tracking, averaging_threshold, False)

    # Draw the lane
    result = drawLane(test_image, warped_image, Minv, plot_y, left_fit_x, right_fit_x, line_tracking, averaging_threshold, False)

    return result

# Lists which store the one time calibration of the camera
objpoints = []
imgpoints = []

# One time call for calibration of the camera
objpoints, imgpoints = cameraCalibration()

# Create a Line object from the Line() class(use default values)
line_tracking = Line()

# Averaging threshold for Phase 3
averaging_threshold = 5

# Video to test on
project_output = 'project_video_output.mp4'
project_clip = VideoFileClip("project_video.mp4")
project_clip = project_clip.fl_image(pipelineVideo)
project_clip.write_videofile(project_output, audio=False)
