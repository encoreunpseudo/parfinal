import cv2
import numpy as np
from scipy.interpolate import interp1d
from .denoise_frame import denoise
def variance_of_laplacian(image):
    
    return cv2.Laplacian(image, cv2.CV_64F).var()

def detect_blur(frame, threshold=100.0):
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    blur_score = variance_of_laplacian(gray)
    
    return blur_score, blur_score < threshold

def interpolate_detections(frame_indices, detections, blur_mask, method='linear'):
   
    frame_indices = np.array(frame_indices)
    
    clear_indices = frame_indices[~blur_mask]
    
    if len(clear_indices) < 2:
        return detections
    
    
    all_ids = set()
    for i in clear_indices:
        for obj_id, _, _ in detections[i].objects_id:
            if obj_id is not None:
                all_ids.add(obj_id)
        
    for obj_id in all_ids:
        obj_frames = []
        obj_boxes = []
            
        for i in clear_indices:
            for tid, cls, box in detections[i].objects_id:
                if tid == obj_id:
                    obj_frames.append(i)
                    obj_boxes.append(box)
                    break
            
        if len(obj_frames) >= 2:
                # Format d'une box: [x1, y1, x2, y2]
            obj_boxes = np.array(obj_boxes)
                
            box_interp = [
                    interp1d(
                        obj_frames, 
                        obj_boxes[:, i], 
                        kind=method, 
                        bounds_error=False, 
                        fill_value="extrapolate"
                    ) for i in range(4)
                ]
                
            for b_idx in blurry_indices:
                if min(obj_frames) <= b_idx <= max(obj_frames):
                    interp_box = [interp(b_idx) for interp in box_interp]
                        
                    interpolated_detections[b_idx].objects_id.append(
                            (obj_id, 'interpolated', interp_box)
                        )
                        
                    interp_conf = float(conf_interp(b_idx))
                    interpolated_detections[b_idx].confidence.append(interp_conf)
                    interpolated_detections[b_idx].interest += 1
    
    return interpolated_detections

def get_batch_frames_with_blur_detection(video_processor, batch_size=8, blur_threshold=100.0, preprocess=False):
   
    cap = cv2.VideoCapture(str(video_processor.path))
    batch_frames = []
    frame_indices = []
    blur_scores = []
    blur_mask = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if batch_frames:
                yield frame_indices, batch_frames, np.array(blur_mask)
            break
                
        if preprocess:
            frame = denoise(frame)
        
        # DETECTION DU FLOU
        blur_score, is_blurry = detect_blur(frame, threshold=blur_threshold)
        blur_scores.append(blur_score)
        blur_mask.append(is_blurry)
        
        batch_frames.append(frame)
        frame_indices.append(frame_idx)
        frame_idx += 1
        
        # When the batch is complete, return the batch
        if len(batch_frames) == batch_size:
            yield frame_indices, batch_frames, np.array(blur_mask)
            batch_frames = []
            frame_indices = []
            blur_mask = []
                
    cap.release()

def process_video_with_blur_detection(self, blur_threshold=100.0):

    total_frames = self.video.frame_count
    frames_processed = 0
    total_blurry = 0
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        for frame_indices, batch_frames, blur_mask in get_batch_frames_with_blur_detection(
            self.video, self.batch_size, blur_threshold, preprocess=True
        ):
            blurry_count = np.sum(blur_mask)
            total_blurry += blurry_count
            
            non_blurry_indices = [idx for idx, blurry in zip(frame_indices, blur_mask) if not blurry]
            non_blurry_frames = [frame for frame, blurry in zip(batch_frames, blur_mask) if not blurry]
            
            if non_blurry_frames:
                # Process faces, hands, and YOLO only for non-blurry frames
                self.process_batch_faces(non_blurry_frames, non_blurry_indices)
                self.process_batch_hands(non_blurry_frames, non_blurry_indices)
                self.process_batch_yolo(non_blurry_frames, non_blurry_indices)
            
            for name, detection in self.detections.items():
                batch_detections = [detection.frames[idx] for idx in frame_indices]
                
                interpolated_detections = interpolate_detections(
                    frame_indices, batch_detections, blur_mask
                )
                
                for i, idx in enumerate(frame_indices):
                    if blur_mask[i]:
                        detection.frames[idx] = interpolated_detections[i]
            
            # Update progress
            frames_processed += len(batch_frames)
            pbar.update(len(batch_frames))
            pbar.set_postfix({
                'blurry': f"{total_blurry/frames_processed:.1%}",
                'processed': frames_processed
            })
    
    logger.info(f"Total frames processed: {frames_processed}")
    logger.info(f"Blurry frames detected: {total_blurry} ({total_blurry/frames_processed:.1%})")
    logger.info(f"Processing time saved: {total_blurry/frames_processed:.1%}")
    
    self.compute_final_statistics()

def calculate_optimal_blur_threshold(video_processor, sample_frames=100):
    
    cap = cv2.VideoCapture(str(video_processor.path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    indices = np.linspace(0, frame_count - 1, sample_frames, dtype=int)
    blur_scores = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            blur_score, _ = detect_blur(frame, threshold=100.0)
            blur_scores.append(blur_score)
    
    cap.release()
    
    if not blur_scores:
        return 100.0
    
   
    optimal_threshold = np.percentile(blur_scores, 30)
    
    return optimal_threshold
