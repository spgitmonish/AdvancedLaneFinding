import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # Line count for averaging
        self.threshold_count = 0

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

        # x values list of last n detected lane pixels(left and right)
        self.all_x = []

        # y values list of last n detected lane pixels(left and right)
        self.all_y = []
