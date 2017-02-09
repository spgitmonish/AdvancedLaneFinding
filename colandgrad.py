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

    # Take the derivative in x
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
