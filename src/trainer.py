"""YOLO model training for helmet detection.

Provides functionality to:
- Train YOLO model on custom dataset
- Validate model performance
- Save trained models
- Generate training reports
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import yaml
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class SafeCityTrainer:
    """Train YOLO model for helmet detection."""
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        project_dir: str = "runs/detect",
        device: str = "cpu",
    ):
        """Initialize trainer.
        
        Args:
            model_name: Base YOLO model (yolov8n, yolov8s, yolov8m, etc.)
            project_dir: Directory to save training results
            device: Device to use ('cpu' or 'cuda')
        """
        if YOLO is None:
            raise RuntimeError("ultralytics package not installed")
        
        self.model_name = model_name
        self.project_dir = Path(project_dir)
        self.device = device
        self.model = None
        self.training_results = None
    
    def load_base_model(self) -> None:
        """Load base YOLO model for training."""
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info(f"Base model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_dataset(
        self,
        data_yaml_path: str,
        validate: bool = True,
    ) -> bool:
        """Validate dataset YAML file.
        
        Args:
            data_yaml_path: Path to data.yaml file
            validate: Whether to validate paths
            
        Returns:
            bool: True if valid
            
        Raises:
            FileNotFoundError: If YAML or data paths not found
        """
        yaml_path = Path(data_yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
        
        # Load YAML
        with open(yaml_path) as f:
            data_config = yaml.safe_load(f)
        
        if validate:
            # Validate paths
            for key in ["train", "val", "test"]:
                if key in data_config:
                    path = Path(data_config[key])
                    if not path.exists():
                        raise FileNotFoundError(
                            f"{key} path not found: {path}"
                        )
        
        logger.info(f"Dataset validated: {yaml_path}")
        return True
    
    def train(
        self,
        data_yaml_path: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        patience: int = 20,
        learning_rate: float = 0.001,
        save_interval: int = 10,
    ) -> Dict[str, Any]:
        """Train model.
        
        Args:
            data_yaml_path: Path to data.yaml
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            patience: Early stopping patience
            learning_rate: Learning rate
            save_interval: Save checkpoint every N epochs
            
        Returns:
            Training results dictionary
            
        Raises:
            RuntimeError: If model not loaded
        """
        if self.model is None:
            self.load_base_model()
        
        # Prepare dataset
        self.prepare_dataset(data_yaml_path)
        
        logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}")
        
        try:
            # Train
            results = self.model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                patience=patience,
                lr0=learning_rate,
                device=self.device,
                project=str(self.project_dir),
                name="helmet_detection",
                exist_ok=True,
                verbose=True,
                save_period=save_interval,
            )
            
            self.training_results = results
            logger.info("Training completed successfully")
            return self._format_results(results)
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate(self, model_path: str, data_yaml_path: str) -> Dict[str, Any]:
        """Validate trained model.
        
        Args:
            model_path: Path to trained model
            data_yaml_path: Path to data.yaml
            
        Returns:
            Validation metrics
        """
        model = YOLO(model_path)
        
        try:
            metrics = model.val(data=data_yaml_path, device=self.device)
            logger.info("Validation completed")
            return self._format_metrics(metrics)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def save_model(
        self,
        output_path: str,
        format: str = "pt",
    ) -> Path:
        """Save trained model.
        
        Args:
            output_path: Path to save model
            format: Model format ('pt', 'onnx', 'tflite', etc.)
            
        Returns:
            Path to saved model
            
        Raises:
            RuntimeError: If model not trained
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "pt":
                self.model.save(str(output_path))
            else:
                self.model.export(format=format)
            
            logger.info(f"Model saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Model info dictionary
        """
        if self.model is None:
            return {"status": "Model not loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.model.parameters()),
            "layers": len(self.model.model),
        }
    
    @staticmethod
    def _format_results(results) -> Dict[str, Any]:
        """Format training results."""
        return {
            "epochs": getattr(results, "epochs", None),
            "best_fitness": getattr(results, "best_fitness", None),
            "save_dir": str(getattr(results, "save_dir", "")),
        }
    
    @staticmethod
    def _format_metrics(metrics) -> Dict[str, Any]:
        """Format validation metrics."""
        return {
            "box_loss": getattr(metrics, "box", None),
            "cls_loss": getattr(metrics, "cls", None),
            "dfl_loss": getattr(metrics, "dfl", None),
            "fitness": getattr(metrics, "fitness", None),
        }


def create_dataset_yaml(
    data_dir: str,
    output_yaml: str = "data.yaml",
    train_split: float = 0.7,
    val_split: float = 0.2,
) -> Path:
    """Create dataset.yaml for YOLO training.
    
    Args:
        data_dir: Root data directory
        output_yaml: Output YAML file path
        train_split: Training data percentage
        val_split: Validation data percentage
        
    Returns:
        Path to created YAML file
    """
    data_dir = Path(data_dir)
    
    data_config = {
        "path": str(data_dir.absolute()),
        "train": str(data_dir / "images/train"),
        "val": str(data_dir / "images/val"),
        "test": str(data_dir / "images/test"),
        "nc": 3,  # Number of classes
        "names": {
            0: "Helmet",
            1: "No_Helmet",
            2: "License_Plate",
        }
    }
    
    output_path = Path(output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    logger.info(f"Dataset YAML created: {output_path}")
    return output_path