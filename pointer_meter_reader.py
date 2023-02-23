import cv2
import numpy as np
from circle_detector import find_circles
from pointer_detector import find_pointers
from utils import compute_angle


def read_meter(image_path):
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess image
    preprocessed = preprocess_image(gray)

    # Find circles in image
    circles = find_circles(preprocessed)

    # Find pointers in image
    pointers = find_pointers(preprocessed, circles)

    # Draw detected pointers and circles on image
    for pointer in pointers:
        cv2.line(img, pointer[0], pointer[1], (0, 0, 255), 2)

    for circle in circles:
        cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

    # Compute angle of each pointer and return results
    results = []
    for i, pointer in enumerate(pointers):
        angle = compute_angle(pointer[0], pointer[1])
        results.append((f"Pointer {i + 1}", angle))
    return results, img


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
