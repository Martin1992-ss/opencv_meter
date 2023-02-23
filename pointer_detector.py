import cv2
import numpy as np

def find_pointers(img, circles):
    pointers = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for circle in circles:
        center_x, center_y, radius = circle
        # crop the pointer region
        pointer_img = gray[int(center_y - 0.7 * radius):int(center_y - 0.3 * radius),
                      int(center_x - 0.15 * radius):int(center_x + 0.15 * radius)]
        # binarize the image
        _, pointer_img = cv2.threshold(pointer_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # find contours in the pointer region
        contours, hierarchy = cv2.findContours(pointer_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # get the largest contour
            max_contour = max(contours, key=cv2.contourArea)
            # compute the centroid of the contour
            moments = cv2.moments(max_contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                # append the pointer coordinates and angle to the pointers list
                pointers.append((cx + int(center_x - 0.15 * radius), cy + int(center_y - 0.7 * radius),
                                  -np.arctan2(moments["mu11"], moments["mu20"]) * 180 / np.pi))
    return pointers
