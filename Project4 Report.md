## Introduction
In Project 1 of this course we students designed a lane finder using Hough Transforms and Canny edge detection. The aim of this project was to do the same but also detect curvature of the line and also the car offset from the center. This project involved more advanced computer vision techniques. Here is a snippet of the pipeline. 

--Insert Pipeline--

## Camera Calibration
First stage of the pipeline was to calibrate the camera using chessboard images. OpenCV has in built functions which takes in chessboard images and searches for the specified number of corners in the images. If the corners were found successfully on the images then the camera is successfully calibrated and images can be undistorted using the camera matrix which has the focal length and camera center information.

The image set provided by Udacity had 9,6 corners. On 3 images in the set I had to use a different set of corners to make sure the OpenCV findChessboardCorners() function detected corners in those images as well. 

> Code: cameracal.py

--Insert Chessboard--

## Distortion Correction 
Once the camera has been calibrated the camera matrix can be used to undistort(radial and tangential) images. A nice way to verify if the camera was calibrated accurately is to verify on a test image like the one shown below. 

--Insert Chessboard undistortion---

The result shows that the undistortion works pretty well. Another check one can do is to apply it on one of the images from the test set.

--Insert road image undistortion--

Although it's not as obvious(when compared to Figure 2), there is distortion correction where needed. 

## Color & Gradient Threshold
In this stage the undistorted image goes through 2 layers of filtering:

1. Color Thresholding: The idea behind this step is to detect the lanes using the properties of an image. The undistorted image is in the RGB format. While RGB is a useful format and various thresholds can be applied(like ranges on R, G, B) to detect lanes, it's not a robust format. There are other formats like HLS and HSV which more closely represent how humans process visual information. 
I had the most success in applying thresholding to detect lanes using the 'S' channel of a HLS image. Even though other areas of the image get highlighted, the area in focus is the road and lanes get highlighted really well. 

--Insert S channel--
2. Gradient Thresholding

