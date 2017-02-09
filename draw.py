import numpy as np
import cv2
import matplotlib.pyplot as plt

def areaInFocus(img, vertices, display_images=False):
    # Convert the 2D Gray image into 3D image with all 1's set to 255
    if img.shape[2]:
        image_in_focus = np.copy(img)
    else:
        image_in_focus = np.dstack((img, img, img))*255

    # Filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.polylines(image_in_focus, np.int32([vertices]), True, (255, 0, 0), 3, 4)

    if display_images == True:
        # Plot the result
        figure4, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        figure4.tight_layout()

        ax1.imshow(img, cmap="gray")
        ax1.set_title('Original Image(S3)', fontsize=20)

        ax2.imshow(image_in_focus, cmap="gray")
        ax2.set_title('Area in focus', fontsize=20)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    return

def drawLane(img, warped, Minv, plot_y, left_fit_x, right_fit_x, line_tracking, averaging_threshold, display_images=False):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    # Calculate the radius based on the equation
    # f(y) = A*y^2 + B*y + C
    # f'(y) = 2A*y + B
    # f''(y) = 2A
    # R(curve) = ((1 + f'(y)^2)^1.5)/f''(y)
    #          = ((1 + (2A*y + B)^2)^1.5)/(2A)

    # Get the maximum value
    result_copy = np.copy(result)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension

    # The top of the car hood
    max_point = (48/720)*30

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(line_tracking.all_y[0]*ym_per_pix, line_tracking.all_x[0]*xm_per_pix, 2)
    right_fit_cr = np.polyfit(line_tracking.all_y[1]*ym_per_pix, line_tracking.all_x[1]*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_radius = ((1 + (2*left_fit_cr[0]*max_point*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_radius = ((1 + (2*right_fit_cr[0]*max_point*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calculate the mean of the two(rounded to 4 decimal places)
    mean_radius = (left_radius + right_radius)/2000
    mean_radius = round(mean_radius, 4)

    if display_images:
        print("LR", left_radius)
        print("RR", right_radius)
        print()

    if(mean_radius >= 2.5):
        radius_display = "Radius of curvature(km): Road is nearly straight"
    else:
        radius_display = "Radius of curvature(km): " + str(mean_radius)

    # Display the text at the top left corner of the image, with hershey simplex,
    # font size as 3, in white, thickness of 2 and line type AA
    cv2.putText(result_copy, str(radius_display), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

    # Calculate the position of the car relative to the road
    car_position = (np.mean(line_tracking.all_x[1]) - np.mean(line_tracking.all_x[0]))/2
    # Convert the car position into meters
    car_position = car_position * xm_per_pix

    # Get the car offset from the center in meters(rounded to 4 decimal places)
    car_offset = car_position - 1.85
    car_offset = round(car_offset, 4)

    # Check where the car is positioned with respect to the center
    if car_offset < 0:
        position_display = "Car is " + str(abs(car_offset)) + "(m) to the left of center"
    else:
        position_display = "Car is " + str(car_offset) + "(m) to the right of center"

    cv2.putText(result_copy, str(position_display), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

    if display_images == True:
        figure9 = plt.figure()
        plt.imshow(result_copy)
        plt.title("Lane Image(mod)")
        plt.show()

    return result_copy
