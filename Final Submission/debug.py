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

def pipelineTestImages(objpoints, imgpoints, line_tracking, averaging_threshold):
    ''' Description: This function tests out various stages of the pipeline
                     on the test images

        Inputs: objpoints - 3d Object points from chessboard corner detection
                imgpoints - 2d Image points from chessboard corner detection
                line_tracking - Object of type Line
                averaging_threshold - Threshold for averaging best fit

        Outputs: None
    '''
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

# Lists which store the one time calibration of the camera
objpoints = []
imgpoints = []

# One time call for calibration of the camera
objpoints, imgpoints = cameraCalibration()

# Create a Line object from the Line() class(use default values)
line_tracking = Line()

# Averaging threshold for Phase 3
averaging_threshold = 3

# Test on images
pipelineTestImages(objpoints, imgpoints, line_tracking, averaging_threshold)
