import cv2 as cv

def select_manual_roi(frame):
    """
    Ouvre une fenêtre pour permettre à l'utilisateur de sélectionner une ROI.
    Retourne (x, y, w, h) ou None si annulé.
    """
    window_name = "Selection de l'objet - Tracez un rectangle et appuyez sur ENTREE"
    
    # Configuration de la fenêtrez
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1280, 720) # Taille confortable

    print(">>> Veuillez sélectionner l'objet avec la souris.")
    
    # Appel de la fonction OpenCV pour sélectionner la ROI
    roi = cv.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    
    cv.destroyWindow(window_name)

    # Déballage et validation
    x, y, w, h = roi

    if w > 0 and h > 0:
        print(f"✅ ROI sélectionnée : {roi}")
        return roi
    else:
        print("❌ Sélection annulée ou invalide.")
        return None