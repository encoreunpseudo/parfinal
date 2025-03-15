import cv2
import numpy as np

def denoise(frame):
    """
    Applique un débruitage adaptatif sur une frame vidéo, optimisé pour les caméras web
    de faible qualité (années 2010).
    
    Paramètres:
    frame (numpy.ndarray): Image d'entrée au format BGR
    
    Retourne:
    numpy.ndarray: Image débruitée
    """
    # Vérification que l'entrée est valide
    if frame is None or frame.size == 0:
        return frame
    
    # Conversion en niveaux de gris pour détecter le niveau de bruit
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
    
    # Estimation du niveau de bruit basée sur la variance locale
    noise_level = np.mean(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Adaptation des paramètres selon le niveau de bruit détecté
    if noise_level < 100:  # Bruit faible
        h_luminance = 3
        h_color = 3
        template_window_size = 7
        search_window_size = 21
    elif noise_level < 500:  # Bruit moyen
        h_luminance = 5
        h_color = 5
        template_window_size = 7
        search_window_size = 21
    else:  # Bruit élevé
        h_luminance = 8
        h_color = 8
        template_window_size = 7
        search_window_size = 35
    
    
    if len(frame.shape) == 3:
        
        denoised = cv2.fastNlMeansDenoisingColored(
            frame, 
            None, 
            h_luminance, 
            h_color, 
            template_window_size, 
            search_window_size
        )
    else:
        
        denoised = cv2.fastNlMeansDenoising(
            frame, 
            None, 
            h_luminance, 
            template_window_size, 
            search_window_size
        )
    
    
    denoised = cv2.bilateralFilter(denoised, 5, 25, 25)
    
    return denoised
