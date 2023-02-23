import cv2
import numpy as np


def preprocess_image(image):
    # Smooth image using Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply adaptive thresholding to obtain binary image
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 3
    )

    # Dilate the thresholded image to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)

    # Apply erosion to remove noise
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded
