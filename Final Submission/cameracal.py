import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def cameraCalibration(display_images=False):
    ''' Description: Calibrates the camera on the chessboard images provided

        Inputs: display_images - When set to True displays images using pyplot

        Outputs: objpoints - 3d Object points from chessboard corner detection
                 imgpoints - 2d Image points from chessboard corner detection
    '''
    # 3D points in real world space
    objpoints = []

    # 2D points in image plane.
    imgpoints = []

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # NOTE: These following combinations are tried to capture the corners on
    #       all the chessboard images provided
    objp_set1 = np.zeros((6*9,3), np.float32)
    objp_set1[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    objp_set2 = np.zeros((6*8,3), np.float32)
    objp_set2[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)
    objp_set3 = np.zeros((5*9,3), np.float32)
    objp_set3[:,:2] = np.mgrid[0:9, 0:5].T.reshape(-1,2)
    objp_set4 = np.zeros((4*9,3), np.float32)
    objp_set4[:,:2] = np.mgrid[0:9, 0:4].T.reshape(-1,2)
    objp_set5 = np.zeros((6*7,3), np.float32)
    objp_set5[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)
    objp_set6 = np.zeros((6*5,3), np.float32)
    objp_set6[:,:2] = np.mgrid[0:5, 0:6].T.reshape(-1,2)

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        # Read the image
        img = mpimg.imread(fname)

        # Convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners using various combinations
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        objp = objp_set1
        if ret == False:
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
            objp = objp_set2
        if ret == False:
            ret, corners = cv2.findChessboardCorners(gray, (9,5), None)
            objp = objp_set3
        if ret == False:
            ret, corners = cv2.findChessboardCorners(gray, (9,4), None)
            objp = objp_set4
        if ret == False:
            ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
            objp = objp_set5
        if ret == False:
            ret, corners = cv2.findChessboardCorners(gray, (5,6), None)
            objp = objp_set6

        # If corners were found, append object points & image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if display_images==True:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (corners.shape[1],corners.shape[0]), corners, ret)
                figure1 = plt.figure()
                plt.imshow(img)
                plt.show()

    # Pick a random image from the list
    img = mpimg.imread('camera_cal/calibration2.jpg')

    # NOTE: img.shape[0] is the width(columns) and img.shape[1] is the height(rows),
    #       img.shape[2] is the number of channels(should be ignored)
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    # NOTE: mtx - Camera matrix
    #       dist - Distorting coeeficients
    #       rvecs - Rotation Vectors for each pattern view
    #       tvecs - Translation Vectors for each pattern view
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Undistort the image using the camera matrix obtained from calibration
    # NOTE: The last parameter is the same as 2nd because the new camera matrix
    #       should be the same as the source
    undistorted_image = cv2.undistort(img, mtx, dist, None, mtx)

    if display_images == True:
        # Plot the result
        figure2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        figure2.tight_layout()

        ax1.imshow(img)
        ax1.set_title('Original', fontsize=20)

        ax2.imshow(undistorted_image)
        ax2.set_title('Undistorted Image', fontsize=20)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    # Return the object points and the image points
    return objpoints, imgpoints

def undistortImage(img, objpoints, imgpoints, display_images=False):
    ''' Description: Undistorts the image using the object and image points

        Inputs: img - Image to be undistorted
                objpoints - 3d Object points from chessboard corner detection
                imgpoints - 2d Image points from chessboard corner detection
                display_images - When set to True displays images using pyplot

        Outputs: undistorted_image - Undistorted image
    '''
    # If the pickle exists load the camera matrix(focal points and camera center)
    # and also load the distortion coefficients to undistort the image
    try:
        dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
    except (OSError, IOError) as e:
        # NOTE: img.shape[0] is the width(columns) and img.shape[1] is the height(rows),
        #       img.shape[2] is the number of channels(should be ignored)
        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        # NOTE: mtx - Camera matrix
        #       dist - Distorting coeeficients
        #       rvecs - Rotation Vectors for each pattern view
        #       tvecs - Translation Vectors for each pattern view
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open("camera_cal/wide_dist_pickle.p", "wb"))

        print("Saving calibration matrix and distortion coefficients")

    # Undistort the image using the camera matrix obtained from calibration
    # NOTE: The last parameter is the same as 2nd because the new camera matrix
    #       should be the same as the source
    undistorted_image = cv2.undistort(img, mtx, dist, None, mtx)

    if display_images == True:
        # Visualize undistortion
        figure3, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        figure3.tight_layout()

        ax1.imshow(img)
        ax1.set_title('Original Image(S1)', fontsize=30)

        ax2.imshow(undistorted_image)
        ax2.set_title('Undistorted Image', fontsize=30)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return undistorted_image
