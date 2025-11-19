import cv2 as cv
import numpy as np

class CamShiftTracker:
    def __init__(self):
        self.roi_hist = None
        self.track_window = None

    def init_camshift(self, frame, roi):
        """
        Initialise CamShift :
        - extrait la ROI
        - convertit en HSV
        - calcule l'histogramme
        """
        x, y, w, h = roi
        self.track_window = (x, y, w, h)

        # ROI dans la frame
        roi_frame = frame[y:y+h, x:x+w]

        # Conversion HSV
        hsv_roi = cv.cvtColor(roi_frame, cv.COLOR_BGR2HSV)

        # Masque (toutes les valeurs valides)
        mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        # Histogramme Hue
        roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])

        # Normalisation
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

        self.roi_hist = roi_hist

    def update_camshift(self, frame):
        """
        Effectue une mise à jour :
        - calcule la backprojection
        - applique CamShift
        - renvoie la frame annotée
        """
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Retourne les zones proches de l'histogramme
        back_proj = cv.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # CamShift
        ret, self.track_window = cv.CamShift(back_proj, self.track_window,
                                             (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1))

        # Rectangle orienté
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        tracked_frame = cv.polylines(frame.copy(), [pts], True, (0, 255, 0), 2)

        return tracked_frame, pts
