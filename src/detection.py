import cv2 as cv

TRACKING_ROI = None 

# 1. Charger la vidéo
capture = cv.VideoCapture(r'C:\Users\XPS\Documents\Object_detection_and_tracking_in_video\data\video1.mp4')

if not capture.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo. Vérifiez le chemin d'accès.")
    exit()

# Lire la première frame pour la sélection
isTrue, frame = capture.read()

if not isTrue:
    print("Erreur : Impossible de lire la première frame.")
    capture.release()
    cv.destroyAllWindows()
    exit()

## 🎯 Sélection Manuelle de la ROI
# Affiche la fenêtre et attend que l'utilisateur dessine un rectangle avec la souris.
#cv.namedWindow("Selection de l'objet - Tracez un rectangle et appuyez sur ENTREE", cv.WINDOW_NORMAL)

# cv.selectROI() retourne les coordonnées (x, y, largeur, hauteur)
roi = cv.selectROI("Selection de l'objet - Tracez un rectangle et appuyez sur ENTREE", 
                   frame, 
                   showCrosshair=True, 
                   fromCenter=False)

cv.destroyWindow("Selection de l'objet - Tracez un rectangle et appuyez sur ENTREE")

# Déballage des coordonnées
x, y, w, h = roi

# Vérifier si une ROI valide a été sélectionnée
if w > 0 and h > 0:
    TRACKING_ROI = (x, y, w, h)
    print(f"✅ ROI sélectionnée (x, y, w, h) : {TRACKING_ROI}")
else:
    print("❌ Sélection annulée ou ROI non valide. Le programme va s'arrêter.")
    capture.release()
    cv.destroyAllWindows()
    exit()


## 📺 Boucle de Lecture et Affichage (Livrable)

while True:
    isTrue, frame = capture.read()
    
    if not isTrue:
        break # Fin de la vidéo

    # Afficher la ROI sélectionnée (validation du livrable)
    x, y, w, h = TRACKING_ROI
    # Dessiner le rectangle sur la frame actuelle
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Vert, épaisseur 2
    
    cv.imshow('Video - ROI Initiale', frame)
    
    # Quitter avec la touche 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
# Nettoyage
capture.release()
cv.destroyAllWindows()