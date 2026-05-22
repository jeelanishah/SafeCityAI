import pytest
from src.data_utils import SplitStats

class TestDataUtils:
    def test_split_stats_creation(self):
        stats = SplitStats(
            train=100, val=20, test=30,
            total=150,
            train_ratio=0.67, val_ratio=0.13, test_ratio=0.2
        )
        assert stats.total == 150
