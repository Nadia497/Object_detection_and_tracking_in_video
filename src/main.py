import cv2 as cv
import utils
import detection
import tracking

def main():
    # --- 1. CHOIX VIDEO ---
    print("--- MENU ---")
    print("1. video.mp4\n2. video2.mp4\n3. video3.mp4\n4. video4.mp4")
    choice = input("Choix vidéo (1-4) : ")
    
    video_map = {"1": "video.mp4", "2": "video2.mp4", "3": "video3.mp4", "4": "video4.mp4"}
    filename = video_map.get(choice, "video.mp4")
    
    # Chargement via utils
    cap = utils.get_video_capture(f"data/{filename}")
    if cap is None: return

    ret, frame = cap.read()
    if not ret:
        print("Erreur lecture première frame")
        return

    # --- 2. CHOIX TRACKER ---
    print("\nQuel algorithme ?")
    print("1. Statique (Juste rectangle, comme ton livrable)")
    print("2. MeanShift")
    print("3. CamShift")
    print("4. Lucas-Kanade")
    
    algo = input("Choix algo (1-4) : ")
    
    if algo == '2': tracker = tracking.MeanShiftTracker()
    elif algo == '3': tracker = tracking.CamShiftTracker()
    elif algo == '4': tracker = tracking.LucasKanadeTracker()
    else:           tracker = tracking.StaticTracker() # Par défaut

    # --- 3. DETECTION / SELECTION (Via detection.py) ---
    # C'est ici qu'on utilise ta logique de selectROI
    roi = detection.select_manual_roi(frame)
    
    if roi is None:
        cap.release()
        return

    # Initialisation du tracker
    tracker.init_tracker(frame, roi)

    # --- 4. BOUCLE PRINCIPALE ---
    cv.namedWindow('Resultat', cv.WINDOW_NORMAL)
    cv.resizeWindow('Resultat', 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mise à jour et affichage
        img_result = tracker.update(frame)
        
        cv.imshow('Resultat', img_result)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()