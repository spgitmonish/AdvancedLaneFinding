import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle

def colorAndGradientThreshold(img, sx_thresh=(20, 120), s_thresh=(120, 220),
                              kernel_size=9, display_images=False):
    ''' Description: Applies color and gradient thresholding to produce a binary
                     image which is used in the next stage of the pipeline

        Inputs: img - Image to be undistorted
                sx_thresh - Threshold range for gradient detection along x-axis
                s_thresh - Threshold range for S channel(HLS) filter
                kernel_size - Sobel Kernel size for gradient detection(odd)
                display_images - When set to True displays images using pyplot

        Outputs: combined_binary - Image after color & gradient thresholding
    '''
    # Convert to HLS color space and separate the channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Determines if gray scale is used or the L channel of an HLS image is used
    use_gray_scale = False

    if use_gray_scale == True:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)
        # Take the derivative in x using gray scale image
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    else:
        # Take the derivative in x using the L channel of the HLS image
        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=kernel_size)

    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobel_x = np.absolute(sobel_x)
    # 8 bit scaling
    scaled_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))

    # Threshold x gradient
    sx_binary = np.zeros_like(scaled_sobel_x)
    sx_binary[(scaled_sobel_x >= sx_thresh[0]) & (scaled_sobel_x <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # NOTE 1: This returns a stack of the two binary images, whose components
    #         you can see as different colors
    # NOTE 2: color_binary[:, :, 0] is all 0s, effectively an all black image.
    #         It might be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sx_binary), sx_binary, s_binary))

    # Combined binary output with color and gradient thresholding
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[((sx_binary==1) | (s_binary==1))] = 1

    if display_images == True:
        # Plot the result
        figure4, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        figure4.tight_layout()

        ax1.imshow(img)
        ax1.set_title('Original Image(S2)', fontsize=20)

        ax2.imshow(combined_binary, cmap="gray")
        ax2.set_title('Color&Gradient Image', fontsize=20)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return combined_binary

def colorAndGradientThresholdNew(img, sx_thresh=(20, 120), h_thresh=(19, 50),
                              s_thresh=(150, 200), kernel_size=9, display_images=False):
    ''' Description: Applies color and gradient thresholding to produce a binary
                     image which is used in the next stage of the pipeline

        Inputs: img - Image to be undistorted
                sx_thresh - Threshold range for gradient detection along x-axis
                s_thresh - Threshold range for S channel(HLS) filter
                kernel_size - Sobel Kernel size for gradient detection(odd)
                display_images - When set to True displays images using pyplot

        Outputs: combined_binary - Image after color & gradient thresholding
    '''
    '''# NOTE: http://softpixel.com/~cwright/programming/colorspace/yuv/
    # Histogram equalization of the image of each channel
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # Equalize the histogram of the Y channel(luminance)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # Convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    if display_images == True:
        figure200 = plt.figure()
        plt.imshow(img_output)
        plt.title("Equalized Image")
        plt.show()

    img = img_output'''

    # Convert image to Lab(Luminance, a&b represent color)
    # NOTE: Here is an explanation http://bit.ly/2lcd2e7
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float)

    # For yellow lines filtering in Lab format
    # 'L' range - (210, 255)
    # 'a' range - (0, 0)
    # 'b' range - (140, 255)
    lab_mask = cv2.inRange(lab, (212, 0, 0), (255, 255, 255))

    '''# For white lines filtering in Lab format
    # 'L' range - (210, 255)
    # 'a' range - (0, 255)
    # 'b' range - (0, 255)
    lab_mask_2 = cv2.inRange(lab, (180, 0, 0), (255, 255, 255))'''

    # Convert image to HSV image
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)

    # For white lines filtering in HSV format
    # NOTE: Here is HSV format visualized http://bit.ly/2lVBxfC
    # 'H' range - (0, 255)
    # 'S' range - (0, 30)
    # 'V' range - (175, 255)
    hsv_mask = cv2.inRange(hsv, (0, 0, 175), (255, 30, 255))

    # Convert image to HLS image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

    # For yellow/white lines filtering in HLS format
    # NOTE: Here is HSL format http://colorizer.org/
    # 'H' range - (0, 130)
    # 'L' range - (0, 255)
    # 'S' range - (84, 255)
    hls_mask = cv2.inRange(hls, (0, 0, 78), (80, 255, 255))

    # Make copies of the image for filtering
    imageCopy1 = np.copy(img)
    imageCopy2 = np.copy(img)
    imageCopy3 = np.copy(img)

    # Lab filtering
    imageCopy1 = cv2.bitwise_and(imageCopy1, imageCopy1, mask=lab_mask)

    # HSV filtering
    imageCopy2 = cv2.bitwise_and(imageCopy2, imageCopy2, mask=hsv_mask)

    # HLS filtering
    imageCopy3 = cv2.bitwise_and(imageCopy3, imageCopy3, mask=hls_mask)

    if display_images == True:
        figure100 = plt.figure()
        plt.imshow(imageCopy1, cmap="gray")
        plt.title("Lab Filtering")
        plt.show()

        figure101 = plt.figure()
        plt.imshow(imageCopy2, cmap="gray")
        plt.title("HSV Filtering")
        plt.show()

        figure102 = plt.figure()
        plt.imshow(imageCopy3, cmap="gray")
        plt.title("HLS Filtering")
        plt.show()

    # Stack each channel to view their individual contributions in green and blue respectively
    # NOTE 1: This returns a stack of the two binary images, whose components
    #         you can see as different colors
    # NOTE 2: color_binary[:, :, 0] is all 0s, effectively an all black image.
    #         It might be beneficial to replace this channel with something else.
    #color_binary = np.dstack((np.zeros_like(l_binary), l_binary, b_binary))

    # Logically OR the masks
    mask = lab_mask | hsv_mask | hls_mask

    # Copy of the image
    imageCopy = np.copy(img)

    # Get the new filtered image
    imageCopy = cv2.bitwise_and(imageCopy, imageCopy, mask=mask)

    # Convert the image to gray scale and return the image
    combined_binary = cv2.cvtColor(imageCopy, cv2.COLOR_RGB2GRAY)

    if display_images == True:
        # Plot the result
        figure4, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        figure4.tight_layout()

        ax1.imshow(img)
        ax1.set_title('Original Image(S2)', fontsize=20)

        ax2.imshow(imageCopy, cmap="gray")
        ax2.set_title('Color Thresholded Image', fontsize=20)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return combined_binary
