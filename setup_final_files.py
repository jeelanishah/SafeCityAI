from pathlib import Path

# F33: README.md
f33 = """# SafeCityAI

Traffic violation detection system using YOLOv5 computer vision.

## Features

- Real-time Detection: Helmet, No-Helmet, License Plate detection
- High Accuracy: YOLOv5 backbone with custom training
- REST API: FastAPI with OpenAPI documentation
- Data Pipeline: Automated augmentation and preprocessing
- Model Training: End-to-end training pipeline
- Testing: Comprehensive pytest coverage
- Dockerized: Ready for production deployment

## Quick Start

git clone https://github.com/yourusername/SafeCityAI.git
cd SafeCityAI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## API Usage

uvicorn api.server:app --reload

## Documentation

- Installation Guide
- API Documentation
- Development Guide

## License

MIT License
"""

# F34: INSTALLATION.md
f34 = """# Installation Guide

## Prerequisites

- Python 3.10+
- 4GB RAM minimum
- 10GB disk space

## Setup

git clone https://github.com/yourusername/SafeCityAI.git
cd SafeCityAI

python -m venv venv
venv\\Scripts\\activate

pip install --upgrade pip
pip install -r requirements.txt

## Environment

cp .env.example .env

## Verify

python -c "from api.server import app; print('Installation successful!')"
"""

# F35: API_DOCUMENTATION.md
f35 = """# API Documentation

## Base URL

http://localhost:8000

## Endpoints

### Health Check

GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}

### Detect Violations

POST /detect

Parameters:
- file (required): Image file
- confidence_threshold (optional): 0.0-1.0

Response:
{
  "success": true,
  "detections": [],
  "num_detections": 1,
  "processing_time_ms": 45.5
}

## Classes

0: Helmet
1: No_Helmet
2: License_Plate

## Interactive Docs

Visit http://localhost:8000/docs
"""

# F36: DEVELOPMENT.md
f36 = """# Development Guide

## Setup

pip install -r requirements-dev.txt

## Testing

pytest tests/ -v

pytest tests/ --cov=src --cov=api

## Code Quality

black src/ api/ tests/

ruff check src/ api/ tests/

mypy src/ api/

## Contributing

1. Create feature branch
2. Make changes
3. Run tests
4. Push and create PR
"""

# F37: Dockerfile
f37 = """FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# F38: docker-compose.yml
f38 = """version: '3.8'

services:
  safecityai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DEBUG=False
      - DEVICE=cpu
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./outputs:/app/outputs
    restart: unless-stopped
"""

# F39: requirements.txt
f39 = """pydantic==2.13.0
pydantic-settings==2.0.3
python-dotenv==1.0.0
fastapi==0.104.1
uvicorn==0.24.0
starlette==0.27.0
torch==2.1.0
torchvision==0.16.0
ultralytics==8.0.200
opencv-python==4.8.1.78
numpy==1.24.3
scikit-learn==1.3.2
albumentations==1.3.1
pillow==10.1.0
requests==2.31.0
tqdm==4.66.1
"""

# F40: requirements-dev.txt
f40 = """pydantic==2.13.0
pydantic-settings==2.0.3
python-dotenv==1.0.0
fastapi==0.104.1
uvicorn==0.24.0
starlette==0.27.0
torch==2.1.0
torchvision==0.16.0
ultralytics==8.0.200
opencv-python==4.8.1.78
numpy==1.24.3
scikit-learn==1.3.2
albumentations==1.3.1
pillow==10.1.0
requests==2.31.0
tqdm==4.66.1
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
httpx==0.25.2
black==23.12.0
ruff==0.1.8
mypy==1.7.1
"""

# F41: .gitignore
f41 = """__pycache__/
*.py[cod]
.venv/
venv/
.env
.env.local
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.idea/
.vscode/
.DS_Store
/data/raw/
/data/processed/
/models/weights/
/outputs/
/logs/
*.log
*.db
*.sqlite
*.pt
*.pth
"""

Path("README.md").write_text(f33, encoding="utf-8")
Path("INSTALLATION.md").write_text(f34, encoding="utf-8")
Path("API_DOCUMENTATION.md").write_text(f35, encoding="utf-8")
Path("DEVELOPMENT.md").write_text(f36, encoding="utf-8")
Path("Dockerfile").write_text(f37, encoding="utf-8")
Path("docker-compose.yml").write_text(f38, encoding="utf-8")
Path("requirements.txt").write_text(f39, encoding="utf-8")
Path("requirements-dev.txt").write_text(f40, encoding="utf-8")
Path(".gitignore").write_text(f41, encoding="utf-8")

print("OK F33-F41 created!")