from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import os
import time
from pathlib import Path
from typing import List

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
MODEL_PATH = "models/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {0: "Helmet", 1: "No_Helmet", 2: "License_Plate"}
VIOLATION_CLASSES = {1, 2}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = None


# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading model...")

    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        print("Model loaded successfully")
    except Exception as e:
        print("Model load failed:", e)
        model = None

    yield
    print("Shutdown")


# ─────────────────────────────
# APP
# ─────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve static UI
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# serve uploads (for output images)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# ─────────────────────────────
# HOME PAGE (OPEN APP HERE)
# ─────────────────────────────
@app.get("/")
def home():
    return FileResponse("static/index.html")


# ─────────────────────────────
# CHECK MODEL
# ─────────────────────────────
def require_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


# ─────────────────────────────
# INFERENCE
# ─────────────────────────────
def run_inference(image):
    start = time.time()
    results = model.predict(image, conf=0.5, verbose=False)
    elapsed = round((time.time() - start) * 1000, 2)

    detections = []
    h = 0
    v = 0

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        name = CLASS_NAMES.get(cls, f"class_{cls}")
        is_violation = cls in VIOLATION_CLASSES

        if is_violation:
            v += 1
        else:
            h += 1

        detections.append({
            "class": name,
            "confidence": round(conf, 3),
            "box": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
            "is_violation": is_violation
        })

    total = len(detections)

    return {
        "total": total,
        "helmets": h,
        "violations": v,
        "detections": detections,
        "time_ms": elapsed
    }


# ─────────────────────────────
# PREDICT SINGLE IMAGE
# ─────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    require_model()

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    result = run_inference(img)

    # save output image with boxes
    try:
        annotated = model.predict(img, conf=0.5, verbose=False)[0].plot()
        out_path = os.path.join(UPLOAD_DIR, f"out_{file.filename}")
        cv2.imwrite(out_path, annotated)

        result["output_image"] = "/uploads/" + os.path.basename(out_path)
    except:
        pass

    return result


# ─────────────────────────────
# OPTIONAL BATCH
# ─────────────────────────────
@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    require_model()

    results = []

    for f in files:
        contents = await f.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            results.append({"file": f.filename, "error": "invalid image"})
            continue

        results.append({"file": f.filename, **run_inference(img)})

    return {"results": results}


# ─────────────────────────────
# RUN (optional)
# ─────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000)