import cv2
import torch
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import json
import logging
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Config:
    MODEL_PATH: str = 'yolov5su.pt'
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    INPUT_VIDEO_PATH: str = 'traffic video dataset.mp4'
    OUTPUT_VIDEO_PATH: str = 'output_video_detected.mp4'
    REPORT_PATH: str = 'detection_report.json'
    LOG_PATH: str = 'video_processing.log'
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_logging(log_path: str):
    logger = logging.getLogger('SafeCityAI')
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging(Config.LOG_PATH)

class SafeCityVideoProcessor:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model = None
        self.logger = logger
        self._load_model()
    
    def _load_model(self):
        self.logger.info(f"Loading model: {self.config.MODEL_PATH}")
        
        if not os.path.exists(self.config.MODEL_PATH):
            self.logger.error(f"Model not found: {self.config.MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {self.config.MODEL_PATH}")
        
        try:
            self.model = YOLO(self.config.MODEL_PATH)
            self.model.to(self.config.DEVICE)
            self.logger.info(f"✅ Model loaded successfully on {self.config.DEVICE}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def process_video(self):
        video_path = self.config.INPUT_VIDEO_PATH
        output_path = self.config.OUTPUT_VIDEO_PATH
        
        self.logger.info(f"Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            self.logger.error(f"Video not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error("Cannot open video file")
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            self.logger.error("Cannot create output video")
            cap.release()
            return None
        
        start_time = time.time()
        frame_count = 0
        total_detections = 0
        helmet_count = 0
        violation_count = 0
        
        self.logger.info("Starting frame processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            results = self.model.predict(
                frame,
                conf=self.config.CONFIDENCE_THRESHOLD,
                verbose=False
            )
            
            frame_detections = len(results[0].boxes)
            total_detections += frame_detections
            
            for box in results[0].boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                if class_id == 0:
                    color = (0, 255, 0)
                    label = f"HELMET {confidence:.2f}"
                    helmet_count += 1
                else:
                    color = (0, 0, 255)
                    label = f"VIOLATION {confidence:.2f}"
                    violation_count += 1
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.putText(frame, f"Frame: {frame_count} | Detections: {frame_detections}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            
            if frame_count % max(1, total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                self.logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        inference_fps = frame_count / processing_time
        
        self.logger.info(f"✅ Processing complete!")
        self.logger.info(f"Output: {output_path}")
        self.logger.info(f"Time: {processing_time:.2f}s")
        self.logger.info(f"Speed: {inference_fps:.2f} FPS")
        self.logger.info(f"Total Detections: {total_detections}")
        self.logger.info(f"Helmets: {helmet_count}")
        self.logger.info(f"Violations: {violation_count}")
        
        report = {
            'video': video_path,
            'total_frames': frame_count,
            'total_detections': total_detections,
            'helmets': helmet_count,
            'violations': violation_count,
            'processing_time_seconds': processing_time,
            'inference_fps': inference_fps,
            'model': self.config.MODEL_PATH,
            'confidence': self.config.CONFIDENCE_THRESHOLD
        }
        
        with open(self.config.REPORT_PATH, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved: {self.config.REPORT_PATH}")
        
        return report

def main():
    try:
        print("\n" + "="*60)
        print("SafeCityAI - Video Processing System")
        print("Helmet Detection & Traffic Violation Analysis")
        print("="*60 + "\n")
        
        config = Config()
        processor = SafeCityVideoProcessor(config)
        report = processor.process_video()
        
        if report:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(f"Total Detections: {report['total_detections']}")
            print(f"Helmets: {report['helmets']}")
            print(f"Violations: {report['violations']}")
            print(f"Processing Time: {report['processing_time_seconds']:.2f}s")
            print(f"Speed: {report['inference_fps']:.2f} FPS")
            print(f"\nOutput Video: {config.OUTPUT_VIDEO_PATH}")
            print(f"Report: {config.REPORT_PATH}")
            print("="*60 + "\n")
            return 0
        else:
            print("Processing failed!")
            return 1
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())