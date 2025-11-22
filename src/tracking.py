import cv2 as cv
import numpy as np
import preprocessing as prep

# --- 1. STATIC TRACKER (Ton code "Livrable simple") ---
class StaticTracker:
    def __init__(self):
        self.roi = None

    def init_tracker(self, frame, roi):
        self.roi = roi # On garde juste les coordonnées fixes

    def update(self, frame):
        # On redessine simplement le rectangle initial
        x, y, w, h = self.roi
        img_result = frame.copy()
        cv.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img_result

# --- 2. MEAN SHIFT ---
class MeanShiftTracker:
    def __init__(self):
        self.roi_hist = None
        self.track_window = None
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    def init_tracker(self, frame, roi):
        self.track_window = roi
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        hsv, mask = prep.get_hsv_mask(roi_frame)
        roi_hist = cv.calcHist([hsv], [0], mask, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        self.roi_hist = roi_hist

    def update(self, frame):
        hsv, _ = prep.get_hsv_mask(frame)
        dst = cv.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        _, self.track_window = cv.meanShift(dst, self.track_window, self.term_crit)
        x, y, w, h = self.track_window
        img_result = cv.rectangle(frame.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img_result

# --- 3. CAM SHIFT ---
class CamShiftTracker:
    def __init__(self):
        self.roi_hist = None
        self.track_window = None
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    def init_tracker(self, frame, roi):
        self.track_window = roi
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        hsv, mask = prep.get_hsv_mask(roi_frame)
        roi_hist = cv.calcHist([hsv], [0], mask, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        self.roi_hist = roi_hist

    def update(self, frame):
        hsv, _ = prep.get_hsv_mask(frame)
        back_proj = cv.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        ret, self.track_window = cv.CamShift(back_proj, self.track_window, self.term_crit)
        pts = cv.boxPoints(ret)
        pts = np.int32(pts)
        return cv.polylines(frame.copy(), [pts], True, (0, 255, 0), 2)

# --- 4. LUCAS KANADE ---
class LucasKanadeTracker:
    def __init__(self):
        self.old_gray = None
        self.points = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    def init_tracker(self, frame, roi):
        x, y, w, h = roi
        self.old_gray = prep.get_gray(frame)
        mask = np.zeros_like(self.old_gray)
        mask[y:y+h, x:x+w] = 255
        self.points = cv.goodFeaturesToTrack(self.old_gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7)

    def update(self, frame):
        frame_gray = prep.get_gray(frame)
        img_result = frame.copy()
        if self.points is None: return img_result

        new_points, status, _ = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.points, None, **self.lk_params)
        
        if new_points is not None:
            good_new = new_points[status == 1]
            for pt in good_new:
                cv.circle(img_result, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            self.old_gray = frame_gray.copy()
            self.points = good_new.reshape(-1, 1, 2)
        return img_result