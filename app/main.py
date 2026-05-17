import os
import sys
import uuid
import datetime
from datetime import timezone
import torch
import torch.nn as nn
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker

# Настройка путей, чтобы Python видел папку ml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- 1. ОЧИСТКА ---
def soft_clean(probs):
    mask = (probs > 0.3).astype(np.uint8) * 255
    if mask.max() == 0:
        return mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# --- 2. БАЗА ---
DATABASE_URL = "sqlite:///./medical_data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnalysisRecord(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    original_path = Column(String)
    mask_path = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 3. ПРИЛОЖЕНИЕ ---
app = FastAPI(title="AI Medical System - Final Fix")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# --- 4. МОДЕЛЬ ---
from app.ml.unet_model import UNet 

device = torch.device("cpu")
model = UNet(n_channels=3, n_classes=1).to(device)
model_path = os.path.join(os.path.dirname(__file__), "ml", "best_model.pth")

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ [AI-SYSTEM] Модель загружена!")
    except Exception as e:
        print(f"❌ [ERROR] Ошибка весов: {e}")

# --- 5. ЭНДПОИНТ ---

@app.post("/upload-dicom")
async def upload_dicom(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        unique_id = str(uuid.uuid4())
        dicom_path = os.path.join(UPLOAD_DIR, f"{unique_id}.dcm")
        with open(dicom_path, "wb") as b:
            b.write(await file.read())

        # Обработка DICOM
        ds = pydicom.dcmread(dicom_path)
        img = apply_voi_lut(ds.pixel_array, ds)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255.0
        img_uint8 = np.uint8(img)

        if len(img_uint8.shape) == 2:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_uint8[:, :, :3]
            
        img_res = cv2.resize(img_rgb, (256, 256))
        
        orig_name = f"{unique_id}_original.jpg"
        cv2.imwrite(os.path.join(UPLOAD_DIR, orig_name), cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR))

        # Инференс
        input_tensor = torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        with torch.no_grad():
            output = model(input_tensor.to(device))
            if isinstance(output, (tuple, list)): output = output[0]
            
            probs = torch.sigmoid(output).squeeze().cpu().numpy()
            max_conf = float(probs.max())
            
            mask_data = soft_clean(probs)
            if mask_data.max() == 0:
                mask_data = (probs > 0.2).astype(np.uint8) * 255

        # Сохранение маски
        mask_name = f"{unique_id}_mask.png"
        cv2.imwrite(os.path.join(UPLOAD_DIR, mask_name), mask_data)

        # Запись в базу
        new_rec = AnalysisRecord(
            filename=file.filename,
            original_path=f"/uploads/{orig_name}",
            mask_path=f"/uploads/{mask_name}"
        )
        db.add(new_rec)
        db.commit()

        print(f"--- [AI RESULT] Confidence: {max_conf:.4f} ---")

        return {
            "status": "success",
            "confidence": round(max_conf, 4),
            "original_image_url": f"http://127.0.0.1:8000/uploads/{orig_name}",
            "ai_mask_url": f"http://127.0.0.1:8000/uploads/{mask_name}"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "online"}