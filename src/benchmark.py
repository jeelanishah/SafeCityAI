"""Performance benchmarking for SafeCityAI.

Provides functionality to:
- Benchmark inference speed
- Measure memory usage
- Profile code performance
- Generate benchmark reports
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
from pathlib import Path
from loguru import logger

try:
    import psutil
except ImportError:
    psutil = None


class InferenceBenchmark:
    """Benchmark inference performance."""
    
    def __init__(self, detector=None):
        """Initialize benchmark.
        
        Args:
            detector: SafeCityDetector instance
        """
        self.detector = detector
        self.results = {}
    
    def benchmark_single_image(
        self,
        image: np.ndarray,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark inference on single image.
        
        Args:
            image: Input image
            num_runs: Number of inference runs
            
        Returns:
            Benchmark results
        """
        if self.detector is None:
            raise RuntimeError("Detector not set")
        
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            _ = self.detector.predict(image)
            elapsed = time.time() - start
            times.append(elapsed)
        
        times = np.array(times)
        
        results = {
            "num_runs": num_runs,
            "mean_time_ms": np.mean(times) * 1000,
            "median_time_ms": np.median(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "fps": 1.0 / np.mean(times),
        }
        
        logger.info(f"Inference benchmark: {results['fps']:.2f} FPS")
        return results
    
    def benchmark_batch(
        self,
        images: List[np.ndarray],
        batch_sizes: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark batch inference.
        
        Args:
            images: List of images
            batch_sizes: Batch sizes to test (default: [1, 4, 8, 16])
            
        Returns:
            Results per batch size
        """
        if self.detector is None:
            raise RuntimeError("Detector not set")
        
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16]
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(images):
                continue
            
            # Prepare batches
            batches = [
                images[i:i + batch_size]
                for i in range(0, len(images), batch_size)
            ]
            
            times = []
            for batch in batches:
                start = time.time()
                _ = self.detector.predict_batch(batch)
                elapsed = time.time() - start
                times.append(elapsed)
            
            times = np.array(times)
            
            results[batch_size] = {
                "mean_time_ms": np.mean(times) * 1000,
                "median_time_ms": np.median(times) * 1000,
                "std_time_ms": np.std(times) * 1000,
                "fps": batch_size / np.mean(times),
            }
            
            logger.info(
                f"Batch size {batch_size}: "
                f"{results[batch_size]['fps']:.2f} FPS"
            )
        
        return results
    
    def benchmark_video(
        self,
        video_path: str,
        frame_interval: int = 1,
    ) -> Dict[str, Any]:
        """Benchmark video processing.
        
        Args:
            video_path: Path to video
            frame_interval: Process every Nth frame
            
        Returns:
            Video processing benchmark results
        """
        if self.detector is None:
            raise RuntimeError("Detector not set")
        
        from src.data_collector import FrameExtractor
        
        try:
            extractor = FrameExtractor(video_path, frame_interval=frame_interval)
            
            start_time = time.time()
            frame_count = 0
            detection_times = []
            
            for frame_num, frame in extractor.extract_frames():
                frame_start = time.time()
                _ = self.detector.predict(frame)
                frame_time = time.time() - frame_start
                detection_times.append(frame_time)
                frame_count += 1
            
            total_time = time.time() - start_time
            detection_times = np.array(detection_times)
            
            results = {
                "video": Path(video_path).name,
                "total_frames": extractor.frame_count,
                "processed_frames": frame_count,
                "fps": extractor.fps,
                "total_time_seconds": total_time,
                "mean_frame_time_ms": np.mean(detection_times) * 1000,
                "throughput_fps": frame_count / total_time,
            }
            
            logger.info(
                f"Video benchmark: {results['throughput_fps']:.2f} FPS "
                f"({frame_count} frames in {total_time:.2f}s)"
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Video benchmark failed: {e}")
            raise


class MemoryBenchmark:
    """Benchmark memory usage."""
    
    def __init__(self, detector=None):
        """Initialize memory benchmark.
        
        Args:
            detector: SafeCityDetector instance
        """
        self.detector = detector
        
        if psutil is None:
            logger.warning("psutil not installed, memory benchmarking disabled")
    
    def get_memory_usage(self) -> Optional[Dict[str, float]]:
        """Get current memory usage.
        
        Returns:
            Memory usage info or None if psutil unavailable
        """
        if psutil is None:
            return None
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident set size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual memory size
        }
    
    def benchmark_memory(self, image: np.ndarray) -> Dict[str, Any]:
        """Benchmark memory usage during inference.
        
        Args:
            image: Input image
            
        Returns:
            Memory usage results
        """
        if psutil is None or self.detector is None:
            return {"status": "Memory benchmarking unavailable"}
        
        # Get memory before
        mem_before = self.get_memory_usage()
        
        # Run inference
        _ = self.detector.predict(image)
        
        # Get memory after
        mem_after = self.get_memory_usage()
        
        return {
            "memory_before_mb": mem_before["rss_mb"],
            "memory_after_mb": mem_after["rss_mb"],
            "memory_increase_mb": mem_after["rss_mb"] - mem_before["rss_mb"],
        }


class ModelBenchmark:
    """Benchmark model properties."""
    
    @staticmethod
    def get_model_size(model_path: str) -> float:
        """Get model file size.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Model size in MB
        """
        path = Path(model_path)
        if not path.exists():
            return 0.0
        
        return path.stat().st_size / 1024 / 1024
    
    @staticmethod
    def get_model_parameters(detector) -> Optional[int]:
        """Get number of model parameters.
        
        Args:
            detector: SafeCityDetector instance
            
        Returns:
            Number of parameters or None
        """
        if not hasattr(detector, 'model') or detector.model is None:
            return None
        
        try:
            total_params = sum(p.numel() for p in detector.model.model.parameters())
            return total_params
        except:
            return None


class BenchmarkReport:
    """Generate comprehensive benchmark report."""
    
    def __init__(self, detector=None):
        """Initialize report generator.
        
        Args:
            detector: SafeCityDetector instance
        """
        self.detector = detector
        self.inference_bench = InferenceBenchmark(detector)
        self.memory_bench = MemoryBenchmark(detector)
        self.model_bench = ModelBenchmark()
    
    def generate_report(
        self,
        test_image: np.ndarray,
        test_images: Optional[List[np.ndarray]] = None,
        num_inference_runs: int = 10,
    ) -> Dict[str, Any]:
        """Generate comprehensive benchmark report.
        
        Args:
            test_image: Single test image
            test_images: List of test images for batch testing
            num_inference_runs: Number of inference runs
            
        Returns:
            Complete benchmark report
        """
        report = {
            "timestamp": time.time(),
            "model": {},
            "inference": {},
            "memory": {},
        }
        
        # Model info
        if self.detector and self.detector.model_path:
            report["model"]["size_mb"] = self.model_bench.get_model_size(
                self.detector.model_path
            )
            report["model"]["parameters"] = self.model_bench.get_model_parameters(
                self.detector
            )
        
        # Single image inference
        report["inference"]["single_image"] = \
            self.inference_bench.benchmark_single_image(
                test_image, num_inference_runs
            )
        
        # Batch inference
        if test_images:
            report["inference"]["batch"] = \
                self.inference_bench.benchmark_batch(test_images)
        
        # Memory usage
        report["memory"]["usage"] = self.memory_bench.benchmark_memory(test_image)
        
        logger.info(f"Benchmark report generated")
        return report
    
    def print_report(self, report: Dict[str, Any]) -> str:
        """Format report as string.
        
        Args:
            report: Benchmark report
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("BENCHMARK REPORT")
        lines.append("=" * 60)
        
        # Model info
        if report.get("model"):
            lines.append("\nMODEL INFO")
            model = report["model"]
            if "size_mb" in model:
                lines.append(f"  Model Size:      {model['size_mb']:.2f} MB")
            if "parameters" in model and model["parameters"]:
                lines.append(
                    f"  Parameters:      {model['parameters']:,}"
                )
        
        # Inference benchmarks
        if report.get("inference"):
            lines.append("\nINFERENCE BENCHMARK")
            if "single_image" in report["inference"]:
                single = report["inference"]["single_image"]
                lines.append(f"  Single Image:")
                lines.append(f"    Mean Time:     {single['mean_time_ms']:.2f} ms")
                lines.append(f"    Median Time:   {single['median_time_ms']:.2f} ms")
                lines.append(f"    FPS:           {single['fps']:.2f}")
            
            if "batch" in report["inference"]:
                lines.append(f"  Batch Processing:")
                for batch_size, metrics in report["inference"]["batch"].items():
                    lines.append(
                        f"    Batch {batch_size}: "
                        f"{metrics['fps']:.2f} FPS"
                    )
        
        # Memory
        if report.get("memory") and report["memory"].get("usage"):
            lines.append("\nMEMORY USAGE")
            mem = report["memory"]["usage"]
            lines.append(
                f"  Before: {mem['memory_before_mb']:.2f} MB"
            )
            lines.append(
                f"  After:  {mem['memory_after_mb']:.2f} MB"
            )
            lines.append(
                f"  Increase: {mem['memory_increase_mb']:.2f} MB"
            )
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)