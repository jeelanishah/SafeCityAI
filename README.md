# SafeCityAI: Production-Grade Traffic Violation Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SafeCityAI is a production-grade computer vision system for automated traffic violation detection using YOLOv5. It detects three key safety violations in real-time:

- **Helmet**: Person wearing safety helmet ✓
- **No_Helmet**: Person on 2-wheeler without helmet ⚠️ VIOLATION
- **License_Plate**: Visible and readable license plate ✓

## Key Features

- 🚀 **Real-time Detection**: Process video streams at 30+ FPS
- 📊 **High Accuracy**: YOLOv5s fine-tuned for traffic scenes (mAP@0.5 > 0.75)
- 🌐 **REST API**: FastAPI endpoints for image and video inference
- 🐳 **Docker Ready**: Production-grade containerization
- 📈 **Monitoring**: Prometheus metrics and Grafana dashboards
- ✅ **Type Safe**: Full type hints with mypy --strict compliance
- 🧪 **Well Tested**: 80%+ code coverage with pytest
- 📚 **Documented**: Google-style docstrings on all modules

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Sources                            │
│  Images  │  Video Files  │  RTSP Streams  │  API Uploads   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Data Preprocessing                         │
│         (Validation, Resizing, Format Conversion)           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  SafeCityAI Detector                         │
│              (YOLOv5s Fine-tuned Model)                     │
│     Helmet (0) │ No_Helmet (1) │ License_Plate (2)         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               Violation Classification                       │
│    Is_Violation: No_Helmet → True, Others → False          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Output Generation                         │
│    REST API Response  │  Annotated Images  │  Metrics      │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- Docker & Docker Compose (optional)

### Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/safecityai/safecityai.git
   cd SafeCityAI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Download model weights**
   ```bash
   python scripts/download_dataset.py
   ```

### Running the API

**Local (Development)**
```bash
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

**Docker (Production)**
```bash
docker-compose -f docker/docker-compose.yml up
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Detect violations in image
curl -X POST http://localhost:8000/api/v1/detect/image \
  -F "file=@traffic_scene.jpg"

# Get model info
curl http://localhost:8000/api/v1/model/info

# Prometheus metrics
curl http://localhost:8000/metrics
```

## Project Structure

```
SafeCityAI/
├── api/                        # FastAPI application
│   ├── config.py              # Configuration management (Pydantic)
│   ├── logger_config.py       # Structured logging (loguru)
│   ├── schemas.py             # Request/response models
│   └── server.py              # FastAPI app and endpoints
├── src/                        # Core ML/DL modules
│   ├── data_collector.py      # Roboflow dataset download
│   ├── annotation_validator.py # YOLO format validation
│   ├── augmentation.py        # Data augmentation pipeline
│   ├── data_utils.py          # Dataset organization (train/val/test)
│   ├── inference.py           # YOLOv5 inference engine
│   └── metrics.py             # Model evaluation metrics
├── tests/                      # Pytest test suite
│   ├── conftest.py            # Pytest fixtures
│   ├── test_api.py            # API endpoint tests
│   └── test_inference.py      # Inference tests
├── scripts/                    # CLI scripts
│   ├── download_dataset.py    # Download from Roboflow
│   ├── train_model.py         # Train YOLOv5
│   ├── evaluate_model.py      # Evaluate model
│   └── run_inference.py       # Run inference
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_demo.ipynb
├── docker/                     # Docker files
│   ├── Dockerfile             # Production image
│   ├── Dockerfile.dev         # Development image
│   └── docker-compose.yml     # Orchestration
├── docs/                       # Documentation
├── datasets/                   # Data storage (gitignored)
├── outputs/                    # Results (gitignored)
└── models/                     # Model weights (gitignored)
```

## Model Details

### Architecture
- **Model**: YOLOv5s (Small variant for edge deployment)
- **Framework**: PyTorch via Ultralytics
- **Input**: 640x640 RGB images (auto-converted from BGR)
- **Output**: Bounding boxes with class probabilities

### Performance (on test set)
- **mAP@0.5**: 0.75+
- **Inference**: 45ms/image (GPU), 200ms/image (CPU)
- **FPS**: 30+ (GPU), 5+ (CPU)
- **Model Size**: 27 MB (for edge devices)

### Classes
| ID | Class | Color | Type | Violation |
|----|-------|-------|------|-----------|
| 0  | Helmet | Green | Positive | No |
| 1  | No_Helmet | Red | Negative | **Yes** |
| 2  | License_Plate | Blue | Reference | No |

## Training

### Dataset Requirements
- Minimum 500 images (aim for 1000+)
- Classes balanced (similar counts per class)
- YOLO format: normalized xywh coordinates in .txt files
- 70/15/15 train/val/test split

### Training Pipeline
```bash
# 1. Collect data
python scripts/download_dataset.py --api-key YOUR_ROBOFLOW_KEY

# 2. Validate and augment
python -m pytest tests/ --cov=src  # Verify data quality

# 3. Train model
python scripts/train_model.py --epochs 100 --batch-size 16

# 4. Evaluate
python scripts/evaluate_model.py --model-path models/weights/best.pt

# 5. Export for production
# (Automatic in training script)
```

## Configuration

All configuration via `.env` file (see `.env.example`):

```env
# Model
MODEL_TYPE=yolov5
MODEL_SIZE=s
CONFIDENCE_THRESHOLD=0.55
IMAGE_SIZE=640

# API
API_HOST=0.0.0.0
API_PORT=8000
MAX_UPLOAD_SIZE_MB=10
API_TIMEOUT_SECONDS=30

# Device
DEVICE=auto  # auto, cuda, mps, cpu

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov=api --cov-report=html

# Run specific test category
pytest tests/ -m unit      # Unit tests only
pytest tests/ -m integration  # Integration tests

# Run with specific loglevel
pytest tests/ --log-cli-level=DEBUG

# Generate coverage badge
coverage-badge -o docs/coverage.svg
```

## Code Quality

```bash
# Linting
ruff check src/ api/ tests/

# Code formatting
black src/ api/ tests/

# Type checking
mypy src/ api/ --strict --ignore-missing-imports

# Pre-commit hooks (recommended)
pre-commit install
pre-commit run --all-files
```

## Deployment

### Docker (Recommended)

```bash
# Build image
docker build -f docker/Dockerfile -t safecityai:latest .

# Run container
docker run -p 8000:8000 --env-file .env safecityai:latest

# Or with docker-compose
docker-compose -f docker/docker-compose.yml up -d
```

### Kubernetes (Production)

See `docs/DEPLOYMENT.md` for Helm charts and K8s manifests.

### Monitoring

- **Prometheus**: Metrics at `http://localhost:9090`
- **Grafana**: Dashboards at `http://localhost:3000`
- **Logs**: Structured JSON logs in `outputs/logs/`

## Performance Optimization

### GPU Acceleration
```python
# Automatic with CUDA 11.8+
# Or explicitly:
export DEVICE=cuda
```

### Batch Processing
```bash
python scripts/run_inference.py --model best.pt --batch-size 32 --video traffic.mp4
```

### Model Optimization
- FP16 inference for 2x speedup (automatic with GPU)
- Model quantization available (INT8, FP16)
- ONNX export for cross-platform deployment

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
export BATCH_SIZE=8
# Or disable GPU
export DEVICE=cpu
```

### Slow Inference
```bash
# Check device
python -c "import torch; print(torch.cuda.is_available())"

# Profile inference
python scripts/run_inference.py --profile --image test.jpg
```

### Missing Dependencies
```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements-dev.txt

# Or with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) file for details.

## Citation

If you use SafeCityAI in your research, please cite:

```bibtex
@software{safecityai2024,
  title={SafeCityAI: Production-Grade Traffic Violation Detection System},
  author={SafeCityAI Team},
  year={2024},
  url={https://github.com/safecityai/safecityai}
}
```

## Support

For issues, questions, or suggestions:
- 📧 Email: support@safecityai.com
- 🐛 GitHub Issues: https://github.com/safecityai/safecityai/issues
- 💬 Discussions: https://github.com/safecityai/safecityai/discussions

## Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the detection architecture
- [Roboflow](https://roboflow.com) for dataset management
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- Community contributors and testers

---

**Last Updated**: 2024
**Version**: 1.0.0
**Status**: Production Ready ✅
