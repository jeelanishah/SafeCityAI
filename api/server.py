from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO

import uvicorn
import shutil
import os
import cv2
import numpy as np

from pathlib import Path

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="SafeCityAI",
    version="1.0.0"
)

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"

STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create folders automatically
STATIC_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# DEBUG LOGS
# =====================================================

print("===================================")
print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("MODEL EXISTS:", MODEL_PATH.exists())

MODELS_FOLDER = BASE_DIR / "models"

if MODELS_FOLDER.exists():
    print("FILES INSIDE models/:")
    print(os.listdir(MODELS_FOLDER))
else:
    print("models folder NOT found")

print("===================================")

# =====================================================
# STATIC FILES
# =====================================================

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# =====================================================
# LOAD YOLO MODEL
# =====================================================

model = None

try:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    print("✅ Model Loaded Successfully")

except Exception as e:

    print("❌ Model Loading Failed")
    print(e)

# =====================================================
# ROOT ROUTE
# =====================================================

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/")
async def home():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        return {
            "message": "SafeCityAI API running successfully"
        }

    return FileResponse(str(index_file))

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
async def health():

    return {
        "status": "running",
        "model_loaded": model is not None,
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH)
    }

# =====================================================
# PREDICT ROUTE
# =====================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="YOLO model not loaded"
        )

    try:

        # Save uploaded image
        file_path = UPLOAD_DIR / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read image
        image = cv2.imread(str(file_path))

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # Convert image to numpy array
        image = np.array(image)

        # YOLO prediction
        results = model.predict(
            source=image,
            conf=0.5,
            save=False
        )

        detections = []

        helmets = 0
        violations = 0

        for result in results:

            boxes = result.boxes

            for box in boxes:

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                coords = box.xyxy[0].tolist()

                is_violation = class_name.lower() in [
                    "no_helmet",
                    "no_seatbelt"
                ]

                if is_violation:
                    violations += 1
                else:
                    helmets += 1

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "box": [round(c, 2) for c in coords],
                    "is_violation": is_violation
                })

        # Save annotated image
        annotated = results[0].plot()

        output_filename = f"detected_{file.filename}"

        output_path = OUTPUT_DIR / output_filename

        cv2.imwrite(str(output_path), annotated)

        return {
            "status": "success",
            "filename": file.filename,
            "total_detections": len(detections),
            "helmets": helmets,
            "violations": violations,
            "detections": detections,
            "output_image_url": f"/output/{output_filename}"
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =====================================================
# SERVE OUTPUT IMAGE
# =====================================================

@app.get("/output/{filename}")
async def get_output_image(filename: str):

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    return FileResponse(str(file_path))

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    PORT = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=PORT
    )