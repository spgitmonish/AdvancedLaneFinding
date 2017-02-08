import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # Line count for averaging
        self.threshold_count = 0

        # Was the line detected in the last iteration?
        self.detected = False

        # Last n fits of the line
        self.recent_fitted = []

        # Average x values of the fitted line over the last n iterations
        self.best_left_x = []
        self.best_right_x = []

        # Polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None
        self.best_fit_right = None

        # Polynomial coefficients for the most recent fit
        self.most_recent_fit = [np.array([False])]

        # Radius of curvature of the line in some units
        self.radius_of_curvature = None

        # Distance in meters of vehicle center from the line
        self.line_base_pos = None

        # Difference in fit coefficients between last and new fits
        self.diff_coefficients = np.array([0,0,0], dtype='float')

        # x values for detected line pixels
        self.all_x = None

        # y values for detected line pixels
        self.all_y = None
