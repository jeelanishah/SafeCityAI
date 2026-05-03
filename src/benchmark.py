"""Performance benchmarking utilities."""

import time
from typing import Dict, List

import numpy as np
from loguru import logger

__all__ = ["Benchmark"]


class Benchmark:
    """Performance benchmarking."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.times = []
        self.memory_usage = []
    
    def start(self) -> float:
        """Start timer."""
        return time.time()
    
    def end(self, start_time: float) -> float:
        """End timer and record time."""
        elapsed = time.time() - start_time
        self.times.append(elapsed)
        return elapsed
    
    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        if not self.times:
            return {}
        
        times = np.array(self.times)
        return {
            'mean': float(times.mean()),
            'std': float(times.std()),
            'min': float(times.min()),
            'max': float(times.max()),
            'median': float(np.median(times)),
            'fps': 1.0 / times.mean(),
        }
    
    def report(self) -> None:
        """Print benchmark report."""
        stats = self.get_stats()
        logger.info("Benchmark Report:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.4f}")
