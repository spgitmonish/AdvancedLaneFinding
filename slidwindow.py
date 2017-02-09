import numpy as np
import cv2
import matplotlib.pyplot as plt

def slidingWindow(img, line_tracking, averaging_threshold, display_images=False):
    ''' Description: The core engine of the pipeline which detects the lines in
                     3 phases -
                     1. Sliding Window
                     2. Search Window around previous search(till threshold)
                     3. Mean Search Window(Low Pass Filter to smooth detection)

        Inputs: img - Image to apply lane detection on
                line_tracking - Object of type 'Line' for tracking detections
                averaging_threshold - Threshold for going from phase 2->3(above)
                display_images - When set to True displays images using pyplot

        Outputs: plot_y - Points along y-axis
                 left_fit_x - Left Lane x-axis points
                 right_fit_x - Right lane x-axis points
                 line_tracking - Modified object with new values
    '''
    # Phase 1
    if np.any(line_tracking.most_recent_fit) == False:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]/4:,:], axis=0)
        if display_images == True:
            figure7 = plt.figure()
            plt.plot(histogram)
            plt.title("Histogram of intensities")
            plt.show()

        # Create an output image to draw on and  visualize the result
        # NOTE: The input image is in gray scale(so only 1 channel)
        output_image = np.dstack((img, img, img))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        left_base_x = np.argmax(histogram[:midpoint])
        right_base_x = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        number_windows = 9

        # Set height of windows
        window_height = np.int(img.shape[0]/number_windows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        # Rows
        nonzero_y = np.array(nonzero[0])
        # Columns
        nonzero_x = np.array(nonzero[1])

        # Current positions to be updated for each window
        left_current_x = left_base_x
        right_current_x = right_base_x

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minimum_pixels = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_indices = []
        right_lane_indices = []

        # Step through the windows one by one
        for window in range(number_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_left_low_x = left_current_x - margin
            win_left_high_x = left_current_x + margin
            win_right_low_x = right_current_x - margin
            win_right_high_x = right_current_x + margin

            # Draw the windows on the visualization image
            cv2.rectangle(output_image,(win_left_low_x,win_y_low),(win_left_high_x,win_y_high),(0,255,0), 2)
            cv2.rectangle(output_image,(win_right_low_x,win_y_low),(win_right_high_x,win_y_high),(0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_indices = ((nonzero_y >= win_y_low) &
                                 (nonzero_y < win_y_high) &
                                 (nonzero_x >= win_left_low_x) &
                                 (nonzero_x < win_left_high_x)).nonzero()[0]

            good_right_indices = ((nonzero_y >= win_y_low) &
                                  (nonzero_y < win_y_high) &
                                  (nonzero_x >= win_right_low_x) &
                                  (nonzero_x < win_right_high_x)).nonzero()[0]

            # Append these indices to the lists
            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)

            # If there were more than 50 pixels found in the window, recenter
            # the window for the next iteration.
            # NOTE: If pixels are leaning left then recenter the window
            #       for the next iteration to the left and vice versa
            if len(good_left_indices) > minimum_pixels:
                left_current_x = np.int(np.mean(nonzero_x[good_left_indices]))
            if len(good_right_indices) > minimum_pixels:
                right_current_x = np.int(np.mean(nonzero_x[good_right_indices]))

        if display_images == True:
            # Plot the result
            figure8, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
            figure8.tight_layout()

            ax1.imshow(img, cmap="gray")
            ax1.set_title('Original Image(S5)', fontsize=20)

            ax2.imshow(output_image)
            ax2.set_title('Sliding Window Image', fontsize=20)

            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()

        # Concatenate the arrays of indices
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        # Extract left and right line pixel positions
        left_x = nonzero_x[left_lane_indices]
        left_y = nonzero_y[left_lane_indices]
        right_x = nonzero_x[right_lane_indices]
        right_y = nonzero_y[right_lane_indices]

        # Fit a second order polynomial for each of the lines
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        # Generate x and y values for plotting
        plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])

        # f(y) = Ay**2 + By + C
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

        # Most recent fit
        line_tracking.most_recent_fit = [left_fit, right_fit]

        # x and y values for detected line pixels
        line_tracking.all_x = [left_x, right_x]
        line_tracking.all_y = [left_y, right_y]

        # Recent coefficients for left and right lanes
        line_tracking.recent_fitted.append([left_fit, right_fit])

        # Increment the count which will be later used for threshold detection
        # for averaging
        line_tracking.threshold_count = line_tracking.threshold_count + 1

        # Draw the lane onto the warped blank image
        if display_images == True:
            # Color the pixels with the indices with red and blue
            output_image[left_y, left_x] = [255, 0, 0]
            output_image[right_y, right_x] = [0, 0, 255]
            figure9 = plt.figure()

            # Plot the x and y values based on the equation
            plt.plot(left_fit_x, plot_y, color='yellow')
            plt.plot(right_fit_x, plot_y, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.imshow(output_image)
            plt.title("Colored Sliding Window Image")

        # Return the values
        return plot_y, left_fit_x, right_fit_x, line_tracking
    # Phase 2
    else:
        # If there is no best fit calculated yet then use the most recent fit
        # for detecting the lanes using search windows from previous detection
        if line_tracking.best_fit_left == None and line_tracking.best_fit_right == None:
            # Line fit for current iteration
            left_fit = np.array([0.0, 0.0, 0.0])
            right_fit = np.array([0.0, 0.0, 0.0])

            # Get the line fit information for left and right lanes
            left_fit[0] = line_tracking.most_recent_fit[0][0]
            left_fit[1] = line_tracking.most_recent_fit[0][1]
            left_fit[2] = line_tracking.most_recent_fit[0][2]

            right_fit[0] = line_tracking.most_recent_fit[1][0]
            right_fit[1] = line_tracking.most_recent_fit[1][1]
            right_fit[2] = line_tracking.most_recent_fit[1][2]

            # Get the non-zero values of the image
            nonzero = img.nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])

            # Width of the window
            margin = 100

            # Find the left and right lane indices within a margin specified
            left_lane_indices = ((nonzero_x > (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y + left_fit[2] - margin)) &
                                 (nonzero_x < (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y + left_fit[2] + margin)))
            right_lane_indices = ((nonzero_x > (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y + right_fit[2] - margin)) &
                                  (nonzero_x < (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y + right_fit[2] + margin)))

            # Extract left and right line pixel positions
            left_x = nonzero_x[left_lane_indices]
            left_y = nonzero_y[left_lane_indices]
            right_x = nonzero_x[right_lane_indices]
            right_y = nonzero_y[right_lane_indices]

            # Fit a second order polynomial to each
            left_fit = np.polyfit(left_y, left_x, 2)
            right_fit = np.polyfit(right_y, right_x, 2)

            # Generate x and y values for plotting
            plot_y = np.linspace(0, img.shape[0]-1, img.shape[0] )
            left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
            right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

            # Most recent fit
            line_tracking.most_recent_fit = [left_fit, right_fit]

            # x and y values for the recent detected line pixels
            line_tracking.all_x = [left_x, right_x]
            line_tracking.all_y = [left_y, right_y]

            # Recent X fits for left and right lanes
            line_tracking.recent_fitted.append([left_fit, right_fit])

            # Increment the count which will be later used for threshold detection
            line_tracking.threshold_count = line_tracking.threshold_count + 1

            # Check to see if the threshold has been met, if so, then compute
            # the average best fit which will get things to Phase 3
            if line_tracking.threshold_count >= averaging_threshold:
                best_fit_left = np.array([0.0, 0.0, 0.0])
                best_fit_right = np.array([0.0, 0.0, 0.0])

                # Go through the list of recent fits
                for index in range(0, len(line_tracking.recent_fitted)):
                    # Get the coefficient values of each of the left and right tracks
                    for coefficient in range(0, 3):
                        best_fit_left[coefficient] = best_fit_left[coefficient] + line_tracking.recent_fitted[index][0][coefficient]
                        best_fit_right[coefficient] = best_fit_right[coefficient] + line_tracking.recent_fitted[index][1][coefficient]

                # Find the average for each of the coefficients
                for coefficient in range(0, 3):
                    best_fit_left[coefficient] = best_fit_left[coefficient]/len(line_tracking.recent_fitted)
                    best_fit_right[coefficient] = best_fit_right[coefficient]/len(line_tracking.recent_fitted)

                # Store the values back into the object
                line_tracking.best_fit_left = best_fit_left
                line_tracking.best_fit_right = best_fit_right

            # Create an image to draw on and an image to show the selection window
            output_image = np.dstack((img, img, img))*255
            window_image = np.zeros_like(output_image)

            # Color in left and right line pixels
            output_image[nonzero_y[left_lane_indices], nonzero_x[left_lane_indices]] = [255, 0, 0]
            output_image[nonzero_y[right_lane_indices], nonzero_x[right_lane_indices]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x-margin, plot_y]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x+margin, plot_y])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))

            right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x-margin, plot_y]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x+margin, plot_y])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            if display_images == True:
                cv2.fillPoly(window_image, np.int_([left_line_pts]), (0,255, 0))
                cv2.fillPoly(window_image, np.int_([right_line_pts]), (0,255, 0))
                result = cv2.addWeighted(output_image, 1, window_image, 0.3, 0)
                figure10 = plt.figure()

                # Plot the lines
                plt.plot(left_fit_x, plot_y, color='yellow')
                plt.plot(right_fit_x, plot_y, color='yellow')
                plt.xlim(0, 1280)
                plt.ylim(720, 0)
                plt.imshow(result)
                plt.title("Sliding Window Image")
                plt.show()

        # Phase 3
        else:
            # Get the non-zero values of the image
            nonzero = img.nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])

            # Width of the window
            margin = 100

            # Line fit for current iteration
            left_fit = np.array([0.0, 0.0, 0.0])
            right_fit = np.array([0.0, 0.0, 0.0])

            # Get the line fit information for left and right lanes
            left_fit[0] = line_tracking.best_fit_left[0]
            left_fit[1] = line_tracking.best_fit_left[1]
            left_fit[2] = line_tracking.best_fit_left[2]

            right_fit[0] = line_tracking.best_fit_right[0]
            right_fit[1] = line_tracking.best_fit_right[1]
            right_fit[2] = line_tracking.best_fit_right[2]

            # Find the left and right lane indices within a margin specified
            left_lane_indices_best = ((nonzero_x > (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y + left_fit[2] - margin)) &
                                     (nonzero_x < (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y + left_fit[2] + margin)))
            right_lane_indices_best = ((nonzero_x > (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y + right_fit[2] - margin)) &
                                      (nonzero_x < (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y + right_fit[2] + margin)))

            # Extract left and right line pixel positions
            left_best_x = nonzero_x[left_lane_indices_best]
            left_best_y = nonzero_y[left_lane_indices_best]
            right_best_x = nonzero_x[right_lane_indices_best]
            right_best_y = nonzero_y[right_lane_indices_best]

            # Get the line fit information for left and right lanes
            left_fit[0] = line_tracking.most_recent_fit[0][0]
            left_fit[1] = line_tracking.most_recent_fit[0][1]
            left_fit[2] = line_tracking.most_recent_fit[0][2]

            right_fit[0] = line_tracking.most_recent_fit[1][0]
            right_fit[1] = line_tracking.most_recent_fit[1][1]
            right_fit[2] = line_tracking.most_recent_fit[1][2]

            # Find the left and right lane indices within a margin specified
            left_lane_indices_recent = ((nonzero_x > (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y + left_fit[2] - margin)) &
                                       (nonzero_x < (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y + left_fit[2] + margin)))
            right_lane_indices_recent = ((nonzero_x > (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y + right_fit[2] - margin)) &
                                        (nonzero_x < (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y + right_fit[2] + margin)))

            # Extract left and right line pixel positions
            left_x = nonzero_x[left_lane_indices_recent]
            left_y = nonzero_y[left_lane_indices_recent]
            right_x = nonzero_x[right_lane_indices_recent]
            right_y = nonzero_y[right_lane_indices_recent]

            # If the number of pixels detected by using the most recent fit is
            # greater than the number detected using the best fit then update
            # values in the Line() object, else don't change the best fit values
            if (len(left_x) > len(left_best_x)) or (len(right_x) > len(right_best_x)):
                # Remove the oldest entry and append new entry
                line_tracking.recent_fitted.pop(0)
                line_tracking.recent_fitted.append([left_fit, right_fit])

                # Update the best fit calculations
                best_fit_left = np.array([0.0, 0.0, 0.0])
                best_fit_right = np.array([0.0, 0.0, 0.0])

                # Go through the list of recent fits
                for index in range(0, len(line_tracking.recent_fitted)):
                    # Get the coefficient values of each of the left and right tracks
                    for coefficient in range(0, 3):
                        best_fit_left[coefficient] = best_fit_left[coefficient] + line_tracking.recent_fitted[index][0][coefficient]
                        best_fit_right[coefficient] = best_fit_right[coefficient] + line_tracking.recent_fitted[index][1][coefficient]

                # Find the average for each of the coefficients
                for coefficient in range(0, 3):
                    best_fit_left[coefficient] = best_fit_left[coefficient]/len(line_tracking.recent_fitted)
                    best_fit_right[coefficient] = best_fit_right[coefficient]/len(line_tracking.recent_fitted)

                # Store the values back into the object
                line_tracking.best_fit_left = best_fit_left
                line_tracking.best_fit_right = best_fit_right

            # Get the line fit information for left and right lanes
            left_fit[0] = line_tracking.best_fit_left[0]
            left_fit[1] = line_tracking.best_fit_left[1]
            left_fit[2] = line_tracking.best_fit_left[2]

            right_fit[0] = line_tracking.best_fit_right[0]
            right_fit[1] = line_tracking.best_fit_right[1]
            right_fit[2] = line_tracking.best_fit_right[2]

            # Find the left and right lane indices within a margin specified
            left_lane_indices = ((nonzero_x > (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y + left_fit[2] - margin)) &
                                (nonzero_x < (left_fit[0]*(nonzero_y**2) + left_fit[1]*nonzero_y + left_fit[2] + margin)))
            right_lane_indices = ((nonzero_x > (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y + right_fit[2] - margin)) &
                                 (nonzero_x < (right_fit[0]*(nonzero_y**2) + right_fit[1]*nonzero_y + right_fit[2] + margin)))

            # Extract left and right line pixel positions
            left_x = nonzero_x[left_lane_indices]
            left_y = nonzero_y[left_lane_indices]
            right_x = nonzero_x[right_lane_indices]
            right_y = nonzero_y[right_lane_indices]

            # Fit a second order polynomial to each
            left_fit = np.polyfit(left_y, left_x, 2)
            right_fit = np.polyfit(right_y, right_x, 2)

            # Generate x and y values for plotting
            plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
            left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
            right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

            # Most recent fit
            line_tracking.most_recent_fit = [left_fit, right_fit]

            # x and y values for the recent detected line pixels
            line_tracking.all_x = [left_x, right_x]
            line_tracking.all_y = [left_y, right_y]

        # Create an image to draw on and an image to show the selection window
        output_image = np.dstack((img, img, img))*255
        window_image = np.zeros_like(output_image)

        # Color in left and right line pixels
        output_image[nonzero_y[left_lane_indices], nonzero_x[left_lane_indices]] = [255, 0, 0]
        output_image[nonzero_y[right_lane_indices], nonzero_x[right_lane_indices]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x-margin, plot_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x+margin, plot_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x-margin, plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x+margin, plot_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        if display_images == True:
            cv2.fillPoly(window_image, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_image, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(output_image, 1, window_image, 0.3, 0)
            figure11 = plt.figure()

            # Plot the lines
            plt.plot(left_fit_x, plot_y, color='yellow')
            plt.plot(right_fit_x, plot_y, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.imshow(result)
            plt.title("Sliding Window Image")
            plt.show()

        return plot_y, left_fit_x, right_fit_x, line_tracking
