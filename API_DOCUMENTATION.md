# API Documentation

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
