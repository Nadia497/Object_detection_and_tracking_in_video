import cv2 as cv
import numpy as np

def get_hsv_mask(frame, low_bound=(0., 60., 32.), high_bound=(180., 255., 255.)):
    """Convertit en HSV et retourne le masque binaire."""
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, np.array(low_bound), np.array(high_bound))
    return hsv, mask

def get_gray(frame):
    """Convertit en niveaux de gris."""
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)