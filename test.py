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

from moviepy.editor import VideoFileClip

def pipelineTestImages(objpoints, imgpoints, line_tracking, averaging_threshold):
    # Undistort the image(needs to go in the pipeline)

    # Make a list of calibration images
    test_images = glob.glob('test_images/straight_lines*.jpg')

    # Set 1
    for idx, fname in enumerate(test_images):
        if(idx == 0) or True:
            # Get the image to test
            test_image = mpimg.imread(fname)

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
            drawLane(test_image, warped_image, Minv, plot_y, left_fit_x, right_fit_x, line_tracking, averaging_threshold, True)

    # Set 2
    test_images = glob.glob('test_images/test*.jpg')
    for idx, fname in enumerate(test_images):
        if(idx == 0) or True:
            # Get the image to test
            test_image = mpimg.imread(fname)

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
            drawLane(test_image, warped_image, Minv, plot_y, left_fit_x, right_fit_x, line_tracking, averaging_threshold, True)

def pipelineVideo(image):
    # The following two variables are globals
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

# Calibrate the camera one time only
objpoints = []
imgpoints = []
objpoints, imgpoints = cameraCalibration()

# Create a Line object from the Line() class(use default values)
line_tracking = Line()

run = 3

# Pipeline test on the images
if run == 1:
    averaging_threshold = 3
    pipelineTestImages(objpoints, imgpoints, line_tracking, averaging_threshold)

elif run == 2:
    averaging_threshold = 10
    def debugRun():
        project_output = 'project_video.mp4'
        project_clip = VideoFileClip("project_video.mp4")
        frameCount = 0
        for frame in project_clip.iter_frames():
            print("\n\nFC: ", str(frameCount))
            #if frameCount == 120:
            result = pipelineVideo(frame)
            frameCount = frameCount + 1

    debugRun()

elif run == 3:
    averaging_threshold = 5
    # Video to test on
    project_output = 'project_video_output.mp4'
    project_clip = VideoFileClip("project_video.mp4")
    project_clip = project_clip.fl_image(pipelineVideo)
    project_clip.write_videofile(project_output, audio=False)
