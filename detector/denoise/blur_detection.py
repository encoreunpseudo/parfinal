import cv2
import numpy as np
from scipy.interpolate import interp1d
from .denoise_frame import denoise
def variance_of_laplacian(image):
    """
    Compute the Laplacian of the image and return the variance.
    A measure of image focus/blur. Lower values indicate more blur.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Variance of the Laplacian response (blur measure)
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def detect_blur(frame, threshold=100.0):
    """
    Detect if a frame is blurry using the Variance of Laplacian method.
    
    Args:
        frame: Input BGR frame
        threshold: Blur threshold (lower means more strict)
        
    Returns:
        blur_score: Numerical blur score (higher means sharper)
        is_blurry: Boolean indicating if the frame is considered blurry
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the variance of Laplacian
    blur_score = variance_of_laplacian(gray)
    
    # Return blur score and boolean result
    return blur_score, blur_score < threshold

def interpolate_detections(frame_indices, detections, blur_mask, method='linear'):
    """
    Interpolate detection data for blurry frames.
    
    Args:
        frame_indices: List of frame indices
        detections: List of detection data for each frame
        blur_mask: Boolean mask where True indicates a blurry frame
        method: Interpolation method ('linear', 'cubic', etc.)
        
    Returns:
        Interpolated detection data
    """
    # Convert to numpy arrays for easier manipulation
    frame_indices = np.array(frame_indices)
    
    # Find indices of non-blurry frames
    clear_indices = frame_indices[~blur_mask]
    
    # If we don't have enough clear frames for interpolation, return original data
    if len(clear_indices) < 2:
        return detections
    
    # Create a copy of the detections to modify
    interpolated_detections = detections.copy()
    
    # For each blurry frame, interpolate the detection data
    blurry_indices = frame_indices[blur_mask]
    
    # We can't directly interpolate the confidence lists since they're of different lengths
    # Instead, we'll focus on interpolating object positions for tracked objects
        
        # Interpolate bounding boxes for each tracked object
        # This is more complex as objects may appear/disappear
        # Here we'll focus on objects that are present in consecutive frames
        
        # Get all unique object IDs
    all_ids = set()
    for i in clear_indices:
        for obj_id, _, _ in detections[i].objects_id:
            if obj_id is not None:
                all_ids.add(obj_id)
        
        # For each object, interpolate its position
    for obj_id in all_ids:
            # Find frames where this object is visible
        obj_frames = []
        obj_boxes = []
            
        for i in clear_indices:
            for tid, cls, box in detections[i].objects_id:
                if tid == obj_id:
                    obj_frames.append(i)
                    obj_boxes.append(box)
                    break
            
            # If object is visible in multiple frames, interpolate
        if len(obj_frames) >= 2:
                # Convert boxes to numpy array for interpolation
                # Box format: [x1, y1, x2, y2]
            obj_boxes = np.array(obj_boxes)
                
                # Create interpolation function for each coordinate
            box_interp = [
                    interp1d(
                        obj_frames, 
                        obj_boxes[:, i], 
                        kind=method, 
                        bounds_error=False, 
                        fill_value="extrapolate"
                    ) for i in range(4)
                ]
                
                # Apply interpolation to blurry frames within the range
            for b_idx in blurry_indices:
                if min(obj_frames) <= b_idx <= max(obj_frames):
                        # Interpolate box coordinates
                    interp_box = [interp(b_idx) for interp in box_interp]
                        
                        # Add interpolated object to the frame
                    interpolated_detections[b_idx].objects_id.append(
                            (obj_id, 'interpolated', interp_box)
                        )
                        
                        # Add interpolated confidence
                    interp_conf = float(conf_interp(b_idx))
                    interpolated_detections[b_idx].confidence.append(interp_conf)
                    interpolated_detections[b_idx].interest += 1
    
    return interpolated_detections

def get_batch_frames_with_blur_detection(video_processor, batch_size=8, blur_threshold=100.0, preprocess=False):
    """
    Generator that returns batches of frames with blur detection.
    
    Args:
        video_processor: VideoProcessor instance
        batch_size: Number of frames per batch
        blur_threshold: Threshold for blur detection
        preprocess: Whether to preprocess frames
        
    Yields:
        frame_indices: List of frame indices
        batch_frames: List of frames
        blur_mask: Boolean mask indicating blurry frames
    """
    cap = cv2.VideoCapture(str(video_processor.path))
    batch_frames = []
    frame_indices = []
    blur_scores = []
    blur_mask = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Process the last batch even if incomplete
            if batch_frames:
                yield frame_indices, batch_frames, np.array(blur_mask)
            break
                
        if preprocess:
            # Apply preprocessing if requested
            frame = denoise(frame)
        
        # Detect blur
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

# Integration into the main VideoProcessor class
def process_video_with_blur_detection(self, blur_threshold=100.0):
    """
    Process the video with blur detection and interpolation.
    """
    total_frames = self.video.frame_count
    frames_processed = 0
    total_blurry = 0
    
    # Use tqdm for progress tracking
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        # Iterate over frame batches with blur detection
        for frame_indices, batch_frames, blur_mask in get_batch_frames_with_blur_detection(
            self.video, self.batch_size, blur_threshold, preprocess=True
        ):
            # Count blurry frames
            blurry_count = np.sum(blur_mask)
            total_blurry += blurry_count
            
            # Process only non-blurry frames to save computation
            non_blurry_indices = [idx for idx, blurry in zip(frame_indices, blur_mask) if not blurry]
            non_blurry_frames = [frame for frame, blurry in zip(batch_frames, blur_mask) if not blurry]
            
            if non_blurry_frames:
                # Process faces, hands, and YOLO only for non-blurry frames
                self.process_batch_faces(non_blurry_frames, non_blurry_indices)
                self.process_batch_hands(non_blurry_frames, non_blurry_indices)
                self.process_batch_yolo(non_blurry_frames, non_blurry_indices)
            
            # For each detection class, interpolate data for blurry frames
            for name, detection in self.detections.items():
                # Extract frames for this batch
                batch_detections = [detection.frames[idx] for idx in frame_indices]
                
                # Interpolate data for blurry frames
                interpolated_detections = interpolate_detections(
                    frame_indices, batch_detections, blur_mask
                )
                
                # Update the detection frames with interpolated data
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
    
    # Log statistics about blur detection
    logger.info(f"Total frames processed: {frames_processed}")
    logger.info(f"Blurry frames detected: {total_blurry} ({total_blurry/frames_processed:.1%})")
    logger.info(f"Processing time saved: {total_blurry/frames_processed:.1%}")
    
    # Compute final statistics
    self.compute_final_statistics()

# Add a method to dynamically calculate the optimal blur threshold
def calculate_optimal_blur_threshold(video_processor, sample_frames=100):
    """
    Calculate an optimal blur threshold based on a sample of frames.
    
    Args:
        video_processor: VideoProcessor instance
        sample_frames: Number of frames to sample
        
    Returns:
        Optimal blur threshold
    """
    cap = cv2.VideoCapture(str(video_processor.path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames uniformly throughout the video
    indices = np.linspace(0, frame_count - 1, sample_frames, dtype=int)
    blur_scores = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            blur_score, _ = detect_blur(frame, threshold=100.0)
            blur_scores.append(blur_score)
    
    cap.release()
    
    # If we couldn't read any frames, return a default value
    if not blur_scores:
        return 100.0
    
    # Calculate threshold using statistics (e.g., percentile-based)
    # Using the 30th percentile as threshold means ~30% of frames will be considered blurry
    optimal_threshold = np.percentile(blur_scores, 30)
    
    return optimal_threshold