# SafeCityAI

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
