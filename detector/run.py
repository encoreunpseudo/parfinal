from main2 import *

video="output.mp4"
detector = BatchDetectionManager(
            video_path=video,
            classes_csv='classes.csv',
            yolo_model='yolov8n.pt'
        )
        
detector.process_video()
        
detector.export_results('results')
        
