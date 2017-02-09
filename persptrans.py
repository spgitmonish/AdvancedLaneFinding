import numpy as np
import cv2
import matplotlib.pyplot as plt
from draw import *

def perspectiveTransform(original, thresholded, display_images=False):
    ''' Description: Applies perspective transform on the image and returns the
                     warped image for lane detection

        Inputs: original - Original Image
                thresholded - Binary Image after color & gradient thresholding
                display_images - When set to True displays images using pyplot

        Outputs: warped_binary - Warped Binary Image
                 Minv - Perspective Transform matrix(destination->source)
    '''
    # Get the image shape
    imshape = original.shape

    # Use the numbers which Udacity has provided
    udacity_set = True

    if udacity_set == True:
        # Source vertices(starting at top-left and going clockwise)
        src_vertices = np.array([(585, 460),
                                 (695, 460),
                                 (1127, 720),
                                 (203, 720)])

        # Display area in focus
        areaInFocus(original, src_vertices, display_images)

        # Destination vertices(starting at top-left and going clockwise)
        dst_vertices = np.array([(200, 0),
                                 (960, 0),
                                 (960, 720),
                                 (200, 720)])
    else:
        # Source vertices(starting at top-left and going clockwise)
        src_vertices = np.array([(600, 440),
                                 (680, 440),
                                 (1140, imshape[0]),
                                 (180, imshape[0])])

        # Display area in focus
        areaInFocus(original, src_vertices, display_images)

        # Destination vertices to get perspectice(starting at top-left and going clockwise)
        dst_vertices = np.array([(200, -500),
                                 (1080, -500),
                                 (1080, imshape[0]),
                                 (200, imshape[0])])

    # Convert the indices into float
    src_vertices = np.float32(src_vertices)
    dst_vertices = np.float32(dst_vertices)

    # Apply the perspective transform and get the matrix
    matrix = cv2.getPerspectiveTransform(src_vertices, dst_vertices)

    # Apply inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst_vertices, src_vertices)

    # Warped image
    # NOTE: Pass in width and height(size of the image)
    warped_image = cv2.warpPerspective(thresholded, matrix, (imshape[1], imshape[0]))
    original_warped = cv2.warpPerspective(original, matrix, (imshape[1], imshape[0]))

    # Display area in focus after warping
    areaInFocus(original_warped, dst_vertices, display_images)

    if display_images == True:
        # Plot the result
        figure6, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        figure6.tight_layout()

        ax1.imshow(thresholded, cmap="gray")
        ax1.set_title('Original Image(S4)', fontsize=20)

        ax2.imshow(warped_image, cmap="gray")
        ax2.set_title('Warped Image', fontsize=20)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return warped_image, Minv
