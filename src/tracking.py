import cv2 as cv
import numpy as np
import preprocessing as prep

# --- 1. COLOR TRACKER  ---
# Dans tracking.py

# Dans tracking.py

class ColorTracker:
    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None
        # Noyau large pour bien fusionner l'objet
        self.kernel = np.ones((11, 11), np.uint8)
        
        self.last_center = None 
        self.last_area = 0

    def init_tracker(self, frame, roi):
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
        hsv_roi = cv.cvtColor(roi_frame, cv.COLOR_BGR2HSV)
        avg_color = np.mean(hsv_roi, axis=(0, 1))
        hue_center = avg_color[0]
        
        # --- CORRECTION DE L'ERREUR ICI ---
        # On ajoute 'dtype=np.uint8' pour forcer le format entier
        self.lower_bound = np.array([max(0, int(hue_center - 25)), 50, 50], dtype=np.uint8)
        self.upper_bound = np.array([min(180, int(hue_center + 25)), 255, 255], dtype=np.uint8)
        
        # Initialisation position et surface
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        self.last_center = (cx, cy)
        self.last_area = w * h

        print(f"✅ Tracker initialisé. Surface initiale : {self.last_area} px")
        print(f"   Seuils (uint8) : {self.lower_bound} -> {self.upper_bound}")

    def update(self, frame):
        img_result = frame.copy()
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # Maintenant ça ne plantera plus ici car les types sont corrects
        mask = cv.inRange(hsv, self.lower_bound, self.upper_bound)
        
        # Nettoyage et fusion
        mask = cv.erode(mask, self.kernel, iterations=1)
        mask = cv.dilate(mask, self.kernel, iterations=3)
        
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        min_dist = float('inf')

        if contours and self.last_center is not None:
            for c in contours:
                area = cv.contourArea(c)
                
                # Filtre : On ignore les objets devenus trop petits (< 30% de la taille originale)
                if area < (self.last_area * 0.3):
                    continue
                
                M = cv.moments(c)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Distance avec la dernière position connue
                dist = np.sqrt((cx - self.last_center[0])**2 + (cy - self.last_center[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_contour = c
            
            if best_contour is not None:
                x, y, w, h = cv.boundingRect(best_contour)
                cv.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv.putText(img_result, "Color Tracker", (x, y-10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Mise à jour
                new_cx = int(x + w / 2)
                new_cy = int(y + h / 2)
                self.last_center = (new_cx, new_cy)
                self.last_area = cv.contourArea(best_contour)
                
                cv.circle(img_result, (new_cx, new_cy), 5, (0, 0, 255), -1)

        return img_result, mask

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