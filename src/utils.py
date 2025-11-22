import cv2 as cv
import os

def get_video_capture(video_path):
    """Charge la vidéo et vérifie si elle est ouverte."""
    # Vérification basique du chemin
    if not os.path.exists(video_path):
        # Tente de corriger le chemin si lancé depuis src/
        if os.path.exists("../" + video_path):
            video_path = "../" + video_path
        else:
            print(f"Erreur : Le fichier {video_path} est introuvable.")
            return None
        
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return None
    return cap