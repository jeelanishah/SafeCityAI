from pathlib import Path

class TrainingCallback:
    def on_epoch_start(self, epoch: int):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: dict):
        pass
    
    def on_training_end(self, best_metrics: dict):
        pass

class EarlyStoppingCallback(TrainingCallback):
    def __init__(self, patience: int = 20, metric: str = "loss"):
        self.patience = patience
        self.metric = metric
        self.best_value = float('inf')
        self.counter = 0
    
    def on_epoch_end(self, epoch: int, metrics: dict):
        if self.metric in metrics:
            current_value = metrics[self.metric]
            if current_value < self.best_value:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")

class CheckpointCallback(TrainingCallback):
    def __init__(self, save_dir: Path, save_every: int = 10):
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, metrics: dict):
        if epoch % self.save_every == 0:
            print(f"Saving checkpoint at epoch {epoch}")
