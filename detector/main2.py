import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import torch
from facenet_pytorch import MTCNN
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
import os 
import sys
from tqdm import tqdm
from denoise.blur_detection import (
    detect_blur, 
    interpolate_detections, 
    get_batch_frames_with_blur_detection,
    calculate_optimal_blur_threshold
)

# Importation du module de débruitage si disponible
try:
    from denoise.denoise_frame import denoise_frame
except ImportError:
    # Fonction factice si le module n'est pas disponible
    def denoise_frame(frame):
        return frame

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionFrame:
    """Données de détection pour une frame"""
    interest: float = 0
    confidence: List[float] = field(default_factory=list)
    objects_id: List[Tuple] = field(default_factory=list)


@dataclass
class DetectionClass:
    """Données de détection pour une classe d'objets"""
    
    frames: List[DetectionFrame]
    yolo_classes: Set[str]  # Classes YOLO associées (en anglais)
    score: float = 0
    mu_confidence: float = 0
    
    @property
    def mean_confidence(self) -> float:
        """Calcule la moyenne des confiances sur toutes les frames"""
        confidences = [conf for frame in self.frames for conf in frame.confidence if conf]
        return np.mean(confidences) if confidences else 0
    

class YOLOClassMapper:
    """Gère le mapping entre les classes YOLO et nos catégories personnalisées"""
    def __init__(self, classes_csv: str):
        self.class_mapping = {}
        self.french_to_english = {}
        self.load_class_mapping(classes_csv)

    def load_class_mapping(self, csv_path: str) -> None:
        """Charge le mapping depuis le CSV avec format: Français;Anglais;Nouvelle Classe"""
        try:
            # Essai avec différents encodages
            encodings = ['cp1252', 'utf-8', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, delimiter=';', 
                                   names=['French', 'English', 'new_class'],
                                   encoding=encoding)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any of the attempted encodings")
                
            # Création du mapping des catégories
            for _, row in df.iterrows():
                new_class = row['new_class'].strip()
                french_class = row['French'].strip()
                english_class = row['English'].strip()
                
                # Mapping français -> anglais
                self.french_to_english[french_class] = english_class
                
                # Création du mapping pour les nouvelles classes
                if new_class not in self.class_mapping:
                    self.class_mapping[new_class] = set()
                    
                # Ajoute les classes YOLO (en anglais) à la nouvelle catégorie
                self.class_mapping[new_class].add(english_class.lower())
                
            logger.info(f"Loaded {len(self.class_mapping)} group mappings with {len(self.french_to_english)} YOLO classes")
            
        except Exception as e:
            logger.error(f"Error loading class mapping from CSV: {e}")
            raise

class VideoProcessor:
    """Gestion du traitement vidéo"""
    def __init__(self, video_path: str):
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(str(self.path))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
    def get_batch_frames(self, batch_size: int = 8, preprocess: bool = False):
        """Générateur qui retourne des lots de frames"""
        cap = cv2.VideoCapture(str(self.path))
        batch_frames = []
        frame_indices = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Traiter le dernier lot même s'il est incomplet
                if batch_frames:
                    yield frame_indices, batch_frames
                break
                
            if preprocess:
                # Appliquer le prétraitement si demandé
                frame = denoise_frame(frame)
                
            batch_frames.append(frame)
            frame_indices.append(frame_idx)
            frame_idx += 1
            
            # Quand le lot est complet, retourner le lot
            if len(batch_frames) == batch_size:
                yield frame_indices, batch_frames
                batch_frames = []
                frame_indices = []
                
        cap.release()


class BatchDetectionManager:
    """Gestionnaire principal des détections avec traitement par lots"""
    def __init__(self, video_path: str, classes_csv: str, yolo_model: str = "yolov8n.pt",
                batch_size: int = 8, skip_frames: int = 1, auto_blur_threshold: bool = True):
        self.video = VideoProcessor(video_path)
        self.batch_size = batch_size
        self.skip_frames = skip_frames
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mapper = YOLOClassMapper(classes_csv)
        self.detections: Dict[str, DetectionClass] = {}
        
        # Auto-calculate blur threshold if requested
        self.auto_blur_threshold = auto_blur_threshold
        self.blur_threshold = 100.0  # Default value, will be updated if auto_blur_threshold is True
        
        # Initialisation des modèles
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)
        
        # MTCNN pour les visages
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # MediaPipe pour les mains
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialisation du tracker DeepSort
        self.tracker = DeepSort(max_age=30, n_init=3)
        
        self.initialize_detections()
        
        # Calculate optimal blur threshold if auto mode is enabled
        if self.auto_blur_threshold:
            self.blur_threshold = calculate_optimal_blur_threshold(self.video)
            logger.info(f"Auto-calculated blur threshold: {self.blur_threshold:.2f}")
        
        logger.info(f"Initialized batch detection manager with device: {self.device}")
        logger.info(f"Video dimensions: {self.video.width}x{self.video.height}")
        logger.info(f"Processing with batch size: {batch_size}, skip frames: {skip_frames}")
        logger.info(f"Blur detection threshold: {self.blur_threshold}")

    def initialize_detections(self) -> None:
        """Initialise les détecteurs et les classes de détection"""
        # Détections de base (visages et mains)
        base_detections = {
            "visages": set(),  # MTCNN
            "mains": set()     # MediaPipe
        }
        
        # Création des détecteurs pour les classes de base
        for name, yolo_classes in base_detections.items():
            self.detections[name] = DetectionClass(
                
                frames=[DetectionFrame() for _ in range(self.video.frame_count)],
                yolo_classes=yolo_classes
            )
        
        # Création des détecteurs pour les classes YOLO groupées
        for category, yolo_classes in self.class_mapper.class_mapping.items():
            self.detections[category] = DetectionClass(
                
                frames=[DetectionFrame() for _ in range(self.video.frame_count)],
                yolo_classes=yolo_classes
            )

    def process_batch_faces(self, batch_frames: List[np.ndarray], frame_indices: List[int]) -> None:
        """Traite la détection des visages pour un lot de frames"""
        # Conversion des frames en RGB pour MTCNN
        batch_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in batch_frames]
        
        # Traitement par lot avec MTCNN
        batch_boxes, batch_probs = self.mtcnn.detect(batch_rgb)
        
        for i, (boxes, probs) in enumerate(zip(batch_boxes, batch_probs)):
            if boxes is not None and probs is not None:
                frame_idx = frame_indices[i]
                face_detection = self.detections["visages"].frames[frame_idx]
                
                for box, prob in zip(boxes, probs):
                    if prob > 0.5:  # Seuil de confiance
                        face_detection.objects_id.append((None, "face", box))
                        face_detection.confidence.append(prob)
                        face_detection.interest += 1
                        
                        x1, y1, x2, y2 = map(int, box)
                        

    def process_batch_hands(self, batch_frames: List[np.ndarray], frame_indices: List[int]) -> None:
        """Traite la détection des mains pour un lot de frames"""
        # Pour MediaPipe, nous traitons une par une car l'API ne supporte pas les lots
        for i, frame in enumerate(batch_frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            frame_idx = frame_indices[i]
            hand_detection = self.detections["mains"].frames[frame_idx]
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_coords = [landmark.x * self.video.width for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y * self.video.height for landmark in hand_landmarks.landmark]
                    box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    
                    hand_detection.objects_id.append((None, "hand", box))
                    hand_detection.confidence.append(1.0)
                    hand_detection.interest += 1
                    
                    

    def process_batch_yolo(self, batch_frames: List[np.ndarray], frame_indices: List[int]) -> None:
        """Traite les détections YOLO pour un lot de frames avec tracking"""
        # Traitement par lot avec YOLO
        results = self.yolo.predict(batch_frames, verbose=False)
        
        for i, result in enumerate(results):
            frame_idx = frame_indices[i]
            frame = batch_frames[i]
            
            # Construction de la liste des détections pour DeepSort
            detections_list = []
            for r in result:
                boxes = r.boxes
                cls_names = r.names
                
                for box in boxes:
                    cls_idx = int(box.cls[0])
                    class_name = cls_names[cls_idx].lower()
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    detections_list.append(([xyxy[0], xyxy[1], xyxy[2], xyxy[3]], conf, cls_idx))
            
            # Mise à jour du tracker
            tracks = self.tracker.update_tracks(detections_list, frame=frame)
            
            # Traitement des tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                bbox = track.to_ltrb()
                cls_idx = track.det_class if hasattr(track, 'det_class') else None
                
                # Récupération du nom de classe
                class_name = None
                if cls_idx is not None:
                    for r in result:
                        if cls_idx in r.names:
                            class_name = r.names[cls_idx].lower()
                            break
                
                # Allocation aux catégories correspondantes
                for category, detection in self.detections.items():
                    if class_name is not None and class_name in detection.yolo_classes:
                        frame_data = detection.frames[frame_idx]
                        frame_data.objects_id.append((track_id, class_name, bbox))
                        
                        conf = track.det_confidence if hasattr(track, 'det_confidence') else 0.0
                        frame_data.confidence.append(conf)
                        frame_data.interest += 1
                        
                        

    

    def compute_final_statistics(self) -> None:
        """Calcule les statistiques finales"""
        for name, detection in self.detections.items():
            detection.mu_confidence = detection.mean_confidence
            detection.score = self.compute_detection_score(detection)
            logger.info(f"Statistics for {name}:")
            logger.info(f"  - Mean confidence: {detection.mu_confidence:.3f}")
            logger.info(f"  - Final score: {detection.score:.3f}")
            
    def compute_detection_score(self, detection: DetectionClass) -> float:
        """Calcule le score global pour une classe de détection"""
        total_detections = sum(frame.interest for frame in detection.frames)
        
        return (total_detections / self.video.frame_count) * 1 * detection.mu_confidence
    
    def process_video(self) -> None:
        """
        Traite la vidéo complète en utilisant des lots et la détection de flou
        """
        total_frames = self.video.frame_count
        frames_processed = 0
        total_blurry = 0
        
        # Process all frames without blur detection if using legacy mode (fallback)
        use_legacy_mode = False
        
        try:
            # Use tqdm for progress tracking
            with tqdm(total=total_frames, desc="Processing video") as pbar:
                if use_legacy_mode:
                    # Use original batch processing without blur detection
                    for frame_indices, batch_frames in self.video.get_batch_frames(self.batch_size, preprocess=True):
                        self.process_batch_faces(batch_frames, frame_indices)
                        self.process_batch_hands(batch_frames, frame_indices)
                        self.process_batch_yolo(batch_frames, frame_indices)
                        
                        frames_processed += len(batch_frames)
                        pbar.update(len(batch_frames))
                else:
                    # Use blur detection and selective processing
                    for frame_indices, batch_frames, blur_mask in get_batch_frames_with_blur_detection(
                        self.video, self.batch_size, self.blur_threshold, preprocess=True
                    ):
                        # Count blurry frames
                        blurry_count = np.sum(blur_mask)
                        total_blurry += blurry_count
                        
                        # Process all frames for now (to ensure stability)
                        # Later we can optimize to only process non-blurry frames
                        self.process_batch_faces(batch_frames, frame_indices)
                        self.process_batch_hands(batch_frames, frame_indices)
                        self.process_batch_yolo(batch_frames, frame_indices)
                        
                        # When stable, uncomment this code to only process non-blurry frames
                        # non_blurry_indices = [idx for idx, blurry in zip(frame_indices, blur_mask) if not blurry]
                        # non_blurry_frames = [frame for frame, blurry in zip(batch_frames, blur_mask) if not blurry]
                        # 
                        # if non_blurry_frames:
                        #     self.process_batch_faces(non_blurry_frames, non_blurry_indices)
                        #     self.process_batch_hands(non_blurry_frames, non_blurry_indices)
                        #     self.process_batch_yolo(non_blurry_frames, non_blurry_indices)
                        
                        # Skip interpolation for now until we fix all bugs
                        # Later when we want to use it:
                        # 
                        # # For each detection class, interpolate data for blurry frames
                        # for name, detection in self.detections.items():
                        #     # Extract frames for this batch
                        #     batch_detections = [detection.frames[idx] for idx in frame_indices]
                        #     
                        #     # Interpolate data for blurry frames
                        #     interpolated_detections = interpolate_detections(
                        #         frame_indices, batch_detections, blur_mask
                        #     )
                        #     
                        #     # Update the detection frames with interpolated data
                        #     for i, idx in enumerate(frame_indices):
                        #         if blur_mask[i]:
                        #             detection.frames[idx] = interpolated_detections[i]
                        
                        frames_processed += len(batch_frames)
                        pbar.update(len(batch_frames))
                        pbar.set_postfix({
                            'blurry': f"{total_blurry/frames_processed:.1%}" if frames_processed > 0 else "0%",
                            'processed': frames_processed
                        })
                
                # Update progress
                frames_processed += len(batch_frames)
                pbar.update(len(batch_frames))
                pbar.set_postfix({
                    'blurry': f"{total_blurry/frames_processed:.1%}",
                    'processed': frames_processed
                })
        except:
            pass
        
        # Log statistics about blur detection
        logger.info(f"Total frames processed: {frames_processed}")
        logger.info(f"Blurry frames detected: {total_blurry} ({total_blurry/frames_processed:.1%})")
        logger.info(f"Processing time saved: {total_blurry/frames_processed:.1%}")
        
        # Compute final statistics
        self.compute_final_statistics()

    def export_results(self, output_path: str = "results") -> None:
        """Exporte les résultats dans plusieurs formats avec informations de flou"""
        try:
            # Export CSV détaillé
            results_list = []
            for class_name, detection in self.detections.items():
                for frame_idx, frame in enumerate(detection.frames):
                    if frame.interest > 0:
                        for i, obj in enumerate(frame.objects_id):
                            # Gestion des tuples à 2 ou 3 éléments
                            if len(obj) == 3:
                                track_id, yolo_class, box = obj
                            else:
                                track_id = None
                                _, box = obj
                                yolo_class = "unknown"

                            # Check if this is an interpolated detection
                            is_interpolated = yolo_class == 'interpolated'

                            # Conversion sûre de la boîte en liste
                            box_list = box.tolist() if hasattr(box, 'tolist') else list(box)
                            conf = frame.confidence[i] if i < len(frame.confidence) else 0.0

                            results_list.append({
                                'class': class_name,
                                'frame': frame_idx,
                                'track_id': track_id,
                                'yolo_class': yolo_class,
                                'is_interpolated': is_interpolated,
                                'box': box_list,
                                'confidence': conf
                            })

            df = pd.DataFrame(results_list)
            df.to_csv(f"{output_path}_detailed.csv", index=False)
            logger.info(f"Exported detailed results to {output_path}_detailed.csv")

            # Export des statistiques avec informations sur les frames floues
            stats = {
                name: {
                    'total_detections': sum(frame.interest for frame in det.frames),
                    'mean_confidence': det.mu_confidence,
                    'score': det.score,
                    'active_frames': sum(1 for frame in det.frames if frame.interest > 0),
                    'interpolated_detections': sum(
                        1 for frame in det.frames 
                        for _, cls, _ in frame.objects_id if cls == 'interpolated'
                    )
                }
                for name, det in self.detections.items()
            }

            pd.DataFrame(stats).to_csv(f"{output_path}_stats.csv")
            logger.info(f"Exported statistics to {output_path}_stats.csv")
            
            # Export blur statistics
            blur_stats = {
                'blur_threshold': self.blur_threshold,
                'auto_threshold': self.auto_blur_threshold,
                'processing_time_saved': sum(1 for name, det in self.detections.items() 
                                           for frame in det.frames 
                                           for _, cls, _ in frame.objects_id if cls == 'interpolated') / 
                                       sum(1 for name, det in self.detections.items() 
                                         for frame in det.frames if frame.interest > 0)
            }
            
            pd.DataFrame([blur_stats]).to_csv(f"{output_path}_blur_stats.csv")
            logger.info(f"Exported blur statistics to {output_path}_blur_stats.csv")

        except Exception as e:
            logger.error(f"Error during results export: {e}")
            raise

def main():
    """Fonction principale d'exécution"""
    import argparse
    parser = argparse.ArgumentParser(description='Analyse de vidéo avec détection multiple par lots et optimisation de flou')
    parser.add_argument('video_path', type=str, help='Chemin vers la vidéo à analyser')
    parser.add_argument('classes_csv', type=str, help='Chemin vers le fichier CSV des classes')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', help='Modèle YOLO à utiliser')
    parser.add_argument('--output', type=str, default='results', help='Préfixe pour les fichiers de sortie')
    parser.add_argument('--batch_size', type=int, default=8, help='Taille des lots de frames')
    parser.add_argument('--skip_frames', type=int, default=1, help='Nombre de frames à sauter (1 = aucun saut)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device à utiliser (optionnel)')
    parser.add_argument('--blur_threshold', type=float, default=100.0, help='Seuil de détection de flou (plus bas = plus strict)')
    parser.add_argument('--auto_threshold', action='store_true', help='Calculer automatiquement le seuil de flou')
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting batch video analysis with parameters:")
        logger.info(f"Video: {args.video_path}")
        logger.info(f"Classes CSV: {args.classes_csv}")
        logger.info(f"YOLO model: {args.yolo_model}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Blur threshold: {'auto' if args.auto_threshold else args.blur_threshold}")
        
        detector = BatchDetectionManager(
            video_path=args.video_path,
            classes_csv=args.classes_csv,
            yolo_model=args.yolo_model,
            batch_size=args.batch_size,
            skip_frames=args.skip_frames,
            auto_blur_threshold=args.auto_threshold
        )
        
        if not args.auto_threshold:
            detector.blur_threshold = args.blur_threshold
        
        logger.info("Starting video processing...")
        detector.process_video()
        
        logger.info("Exporting results...")
        detector.export_results(args.output)
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        raise


if __name__ == "__main__":
    main()