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
        if(idx < 0) and True:
            # Get the image to test
            test_image = mpimg.imread(fname)

            test_image_size = test_image.shape

            # If there is an alpha channel convert to RGB only
            if(test_image_size[2] > 3):
                test_image = cv2.cvtColor(test_image, cv2.COLOR_RGBA2RGB)

            # Undistort the image
            undistorted_image = undistortImage(test_image, objpoints, imgpoints, False)

            # Apply color and gradient thresholding
            thresholded_image = colorAndGradientThreshold(img=undistorted_image, display_images=True)

            # Get the perspective transform of the image
            warped_image, Minv = perspectiveTransform(test_image, thresholded_image, True)
            warped = np.copy(warped_image)

            # Apply sliding window
            plot_y, left_fit_x, right_fit_x, line_tracking = slidingWindow(warped, line_tracking, averaging_threshold, True)

            # Draw the lane
            drawLane(test_image, warped_image, Minv, plot_y, left_fit_x, right_fit_x, line_tracking, averaging_threshold, True)

    # Set 2
    test_images = glob.glob('test_images/test*.jpg')
    for idx, fname in enumerate(test_images):
        if(idx < 0) and True:
            # Get the image to test
            test_image = mpimg.imread(fname)

            test_image_size = test_image.shape

            # If there is an alpha channel convert to RGB only
            if(test_image_size[2] > 3):
                test_image = cv2.cvtColor(test_image, cv2.COLOR_RGBA2RGB)

            # Undistort the image
            undistorted_image = undistortImage(test_image, objpoints, imgpoints, False)

            # Apply color and gradient thresholding
            thresholded_image = colorAndGradientThreshold(img=undistorted_image, display_images=True)

            # Get the perspective transform of the image
            warped_image, Minv = perspectiveTransform(test_image, thresholded_image, True)
            warped = np.copy(warped_image)

            # Apply sliding window
            plot_y, left_fit_x, right_fit_x, line_tracking = slidingWindow(warped, line_tracking, averaging_threshold, True)

            # Draw the lane
            drawLane(test_image, warped_image, Minv, plot_y, left_fit_x, right_fit_x, line_tracking, averaging_threshold, True)

    # Set 3
    test_images = glob.glob('video_images/FC*.jpg')
    for idx, fname in enumerate(test_images):
        #if((idx > 30) and (idx < 35)) and True:
        if fname == "video_images\FC120.jpg" or fname == "video_images\FC122.jpg" or fname == "video_images\FC124.jpg":
            # Get the image to test
            test_image = mpimg.imread(fname)

            test_image_size = test_image.shape

            # If there is an alpha channel convert to RGB only
            if(test_image_size[2] > 3):
                test_image = cv2.cvtColor(test_image, cv2.COLOR_RGBA2RGB)

            # Undistort the image
            undistorted_image = undistortImage(test_image, objpoints, imgpoints, False)

            # Apply color and gradient thresholding
            thresholded_image = colorAndGradientThreshold(img=undistorted_image, display_images=True)

            # Get the perspective transform of the image
            warped_image, Minv = perspectiveTransform(test_image, thresholded_image, True)
            warped = np.copy(warped_image)

            # Apply sliding window
            plot_y, left_fit_x, right_fit_x, line_tracking = slidingWindow(warped, line_tracking, averaging_threshold, True)

            # Draw the lane
            drawLane(test_image, warped_image, Minv, plot_y, left_fit_x, right_fit_x, line_tracking, averaging_threshold, True)

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
averaging_threshold = 3
def debugRun():
    #project_clip = VideoFileClip("project_video.mp4")
    project_clip = VideoFileClip("challenge_video.mp4")
    frameCount = 0
    for frame in project_clip.iter_frames():
        #print("FC: " + str(frameCount) + ", RoC: " + str(line_tracking.radius_of_curvature))
        print("FC: " + str(frameCount))
        filename = "video_images_output/FC" + str(frameCount) + ".jpg"
        result = pipelineVideo(frame)
        # Save file
        mpimg.imsave(filename, result)
        frameCount = frameCount + 1

#debugRun()

# Test on images
pipelineTestImages(objpoints, imgpoints, line_tracking, averaging_threshold)
