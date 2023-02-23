import cv2


def find_circles(image):
    # Detect circles in image using HoughCircles transform
    circles = cv2.HoughCircles(
        image, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=100, param2=50, minRadius=50, maxRadius=300
    )

    # Convert coordinates to integers
    circles = np.round(circles[0, :]).astype("int")

    return circles
