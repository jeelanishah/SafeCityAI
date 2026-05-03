# 🚨 SafeCityAI - Traffic Violation Detection System

Production-grade YOLOv5-based real-time traffic violation detection system with helmet compliance, license plate detection, and intelligent alert mechanisms.

## Features

- Real-time Detection: YOLOv5-based detection with <100ms latency
- Multi-class Detection: Helmet, No-Helmet, License Plate
- REST API: FastAPI with automatic documentation
- Docker Ready: Production-grade containers
- Monitoring: Prometheus metrics
- CI/CD: GitHub Actions automation
- Type Safe: Full type hints and validation
- Comprehensive Tests: 80%+ code coverage

## Tech Stack

- Framework: FastAPI + Uvicorn
- ML/DL: PyTorch + YOLO v5
- Data: OpenCV + Albumentations + NumPy
- Validation: Pydantic v2
- Logging: Loguru
- Testing: Pytest + Coverage
- Containerization: Docker + Docker Compose
- CI/CD: GitHub Actions
- Monitoring: Prometheus

## Installation

### Prerequisites
- Python 3.10+
- Git
- Conda or pip

### Step 1: Clone Repository

git clone https://github.com/jeelanishahi/SafeCityAI.git
cd SafeCityAI

### Step 2: Create Environment

conda create -n safecityai python=3.10 -y
conda activate safecityai

### Step 3: Install Dependencies

pip install -r requirements.txt

### Step 4: Verify Installation

python -c "from api.config import settings; print('Installation successful!')"

## Quick Start

### Start API Server

uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Server will be available at: http://localhost:8000

### API Endpoints

Health Check:
curl http://localhost:8000/health

Run Inference:
curl -X POST http://localhost:8000/predict -F "file=@test_image.jpg"

List Models:
curl http://localhost:8000/models

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker Deployment

### Development

cd docker
docker-compose up

### Production

docker build -f docker/Dockerfile -t safecityai:latest .
docker run -p 8000:8000 safecityai:latest

## Configuration

Edit .env file to customize:

ENVIRONMENT=production
API_PORT=8000
IMAGE_SIZE=640
CONF_THRESHOLD=0.55
DEVICE=auto
LOG_LEVEL=INFO

## Training

jupyter notebook notebooks/02_training.ipynb

## Testing

pytest tests/ -v --cov=api --cov=src

## Performance

- Inference Time: ~42ms
- Model Size: ~87MB
- Accuracy (mAP@0.5): 0.92
- Throughput: ~24 FPS

## Author

Abdul Khader Jeelani Fayaz
Email: abdul.23aiml@cambridge.edu.in
GitHub: @jeelanishahi

## License

MIT License
