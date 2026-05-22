"""Video processing and frame extraction for helmet detection.

Provides functionality to:
- Extract frames from videos
- Process video streams
- Handle batch frame processing
- Generate frame statistics
"""

from pathlib import Path
from typing import Optional, List, Tuple, Generator, Dict, Any
import cv2
import numpy as np
from loguru import logger


class FrameExtractor:
    """Extract frames from video files."""
    
    def __init__(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        frame_interval: int = 1,
    ):
        """Initialize frame extractor.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame (1 = all frames)
            
        Raises:
            FileNotFoundError: If video not found
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else None
        self.frame_interval = frame_interval
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        # Create output directory if specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video properties
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(
            f"Video loaded: {self.frame_count} frames @ {self.fps} fps "
            f"({self.width}x{self.height})"
        )
    
    def get_video_info(self) -> Dict[str, Any]:
        """Get video information.
        
        Returns:
            Dictionary with video properties
        """
        return {
            "filename": self.video_path.name,
            "total_frames": self.frame_count,
            "fps": self.fps,
            "duration_seconds": self.frame_count / self.fps,
            "width": self.width,
            "height": self.height,
            "frame_interval": self.frame_interval,
        }
    
    def extract_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Extract frames from video.
        
        Yields:
            Tuple of (frame_number, frame_array)
        """
        frame_num = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                if frame_num % self.frame_interval == 0:
                    extracted_count += 1
                    yield frame_num, frame
                
                frame_num += 1
        
        finally:
            self.cap.release()
            logger.info(f"Extraction complete: {extracted_count} frames extracted")
    
    def extract_and_save_frames(
        self,
        format: str = "jpg",
    ) -> List[Path]:
        """Extract and save frames to disk.
        
        Args:
            format: Image format ('jpg', 'png', etc.)
            
        Returns:
            List of saved frame paths
            
        Raises:
            ValueError: If output_dir not set
        """
        if not self.output_dir:
            raise ValueError("output_dir not set")
        
        saved_frames = []
        
        for frame_num, frame in self.extract_frames():
            filename = f"frame_{frame_num:06d}.{format}"
            filepath = self.output_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            saved_frames.append(filepath)
        
        logger.info(f"Frames saved to: {self.output_dir}")
        return saved_frames
    
    def get_frame_at(self, frame_num: int) -> Optional[np.ndarray]:
        """Get specific frame.
        
        Args:
            frame_num: Frame number to retrieve
            
        Returns:
            Frame array or None if invalid
        """
        if frame_num < 0 or frame_num >= self.frame_count:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def get_frame_at_time(self, seconds: float) -> Optional[np.ndarray]:
        """Get frame at specific time.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Frame array or None if invalid
        """
        frame_num = int(seconds * self.fps)
        return self.get_frame_at(frame_num)
    
    def extract_key_frames(self, num_frames: int = 10) -> List[np.ndarray]:
        """Extract evenly spaced key frames.
        
        Args:
            num_frames: Number of key frames to extract
            
        Returns:
            List of key frames
        """
        key_frames = []
        interval = self.frame_count // (num_frames + 1)
        
        for i in range(1, num_frames + 1):
            frame_num = i * interval
            frame = self.get_frame_at(frame_num)
            if frame is not None:
                key_frames.append(frame)
        
        return key_frames
    
    def __del__(self):
        """Cleanup video capture."""
        if hasattr(self, 'cap'):
            self.cap.release()


class VideoProcessor:
    """Process video frames with detector."""
    
    def __init__(
        self,
        video_path: str,
        detector=None,
        output_video_path: Optional[str] = None,
        frame_interval: int = 1,
    ):
        """Initialize video processor.
        
        Args:
            video_path: Path to input video
            detector: SafeCityDetector instance
            output_video_path: Path to save output video with annotations
            frame_interval: Process every Nth frame
        """
        self.extractor = FrameExtractor(video_path, frame_interval=frame_interval)
        self.detector = detector
        self.output_video_path = output_video_path
        self.processed_count = 0
        self.violations_count = 0
        self.violations = []
    
    def process_video(self, save_annotated: bool = False) -> Dict[str, Any]:
        """Process entire video.
        
        Args:
            save_annotated: Whether to save annotated video
            
        Returns:
            Processing results dictionary
        """
        if not self.detector:
            raise RuntimeError("Detector not set")
        
        logger.info(f"Processing video: {self.extractor.video_path.name}")
        
        # Setup video writer if saving
        writer = None
        if save_annotated and self.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(self.output_video_path),
                fourcc,
                self.extractor.fps,
                (self.extractor.width, self.extractor.height),
            )
        
        try:
            for frame_num, frame in self.extractor.extract_frames():
                # Run detection
                detections = self.detector.predict(frame)
                
                # Extract violations
                violations = self.detector.get_violations(detections)
                if violations:
                    self.violations_count += len(violations)
                    self.violations.append({
                        "frame": frame_num,
                        "time_seconds": frame_num / self.extractor.fps,
                        "violations": violations,
                    })
                
                # Annotate if saving
                if writer:
                    annotated = self.detector.annotate_image(frame, detections)
                    writer.write(annotated)
                
                self.processed_count += 1
                
                # Log progress
                if self.processed_count % 30 == 0:
                    logger.info(f"Processed {self.processed_count} frames")
        
        finally:
            if writer:
                writer.release()
                logger.info(f"Annotated video saved: {self.output_video_path}")
        
        return self._get_results()
    
    def process_frames_batch(
        self,
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """Process video in batches.
        
        Args:
            batch_size: Number of frames per batch
            
        Returns:
            List of batch results
        """
        if not self.detector:
            raise RuntimeError("Detector not set")
        
        batch_results = []
        batch_frames = []
        batch_frame_nums = []
        
        for frame_num, frame in self.extractor.extract_frames():
            batch_frames.append(frame)
            batch_frame_nums.append(frame_num)
            
            if len(batch_frames) == batch_size:
                # Process batch
                results = self.detector.predict_batch(batch_frames)
                
                for fn, detections in zip(batch_frame_nums, results):
                    violations = self.detector.get_violations(detections)
                    batch_results.append({
                        "frame": fn,
                        "detections": len(detections),
                        "violations": len(violations),
                    })
                    
                    if violations:
                        self.violations_count += len(violations)
                
                batch_frames = []
                batch_frame_nums = []
                self.processed_count += batch_size
        
        # Process remaining frames
        if batch_frames:
            results = self.detector.predict_batch(batch_frames)
            for fn, detections in zip(batch_frame_nums, results):
                violations = self.detector.get_violations(detections)
                batch_results.append({
                    "frame": fn,
                    "detections": len(detections),
                    "violations": len(violations),
                })
                if violations:
                    self.violations_count += len(violations)
            
            self.processed_count += len(batch_frames)
        
        return batch_results
    
    def _get_results(self) -> Dict[str, Any]:
        """Get processing results."""
        return {
            "video": self.extractor.video_path.name,
            "total_frames": self.extractor.frame_count,
            "processed_frames": self.processed_count,
            "fps": self.extractor.fps,
            "duration_seconds": self.extractor.frame_count / self.extractor.fps,
            "total_violations": self.violations_count,
            "violations_details": self.violations,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        return self._get_results()