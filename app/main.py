import os
import uuid
import datetime
from datetime import timezone
import torch
import torch.nn as nn
import pydicom
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker

# --- 1. БАЗА ДАННЫХ ---
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
    try: yield db
    finally: db.close()

# --- 2. ПРИЛОЖЕНИЕ ---
app = FastAPI(title="AI Medical API - Production Ready")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# --- 3. ИНИЦИАЛИЗАЦИЯ ИИ-МОДЕЛИ ---
from app.ml.unet_model import UNet # Импорт твоего класса из unet_model.py

device = torch.device("cpu")
model = UNet(n_channels=3, n_classes=1) # Теперь n_channels=3 по умолчанию
model_path = "app/ml/best_model.pth"

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ [AI-SYSTEM] Модель синхронизирована (3 канала)!")
    except Exception as e:
        print(f"❌ [CRITICAL] Ошибка весов: {e}")

# --- 4. ЭНДПОИНТ ОБРАБОТКИ ---

@app.post("/upload-dicom")
async def upload_dicom(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        unique_id = str(uuid.uuid4())
        
        # Шаг 1: Сохранение исходного DICOM
        dicom_path = os.path.join(UPLOAD_DIR, f"{unique_id}.dcm")
        with open(dicom_path, "wb") as b:
            b.write(await file.read())

        # Шаг 2: Обработка снимка через OpenCV
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(float)
        
        # Нормализация 0-255
        img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255.0
        img_uint8 = np.uint8(img)

        # "Умная" проверка каналов (исправление ошибки cvtColor)
        if len(img_uint8.shape) == 2:
            # Если снимок ЧБ, делаем из него 3 канала
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        else:
            # Если уже 3 канала, берем как есть
            img_rgb = img_uint8[:, :, :3]

        img_res = cv2.resize(img_rgb, (256, 256))
        
        # Сохраняем оригинал как JPG (врачу для просмотра)
        orig_name = f"{unique_id}_original.jpg"
        cv2.imwrite(os.path.join(UPLOAD_DIR, orig_name), cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR))

        # Шаг 3: Инференс (как в твоем test_ai.py)
        # Переводим в тензор, меняем оси (HWC -> CHW) и нормируем на 255
        input_tensor = torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        mask_name = "no_mask.png"
        confidence = 0.0

        with torch.no_grad():
            output = model(input_tensor.to(device))
            # Обработка вывода (если вдруг модель вернет кортеж)
            if isinstance(output, (tuple, list)): output = output[0]
            
            probs = torch.sigmoid(output).cpu().numpy()[0][0]
            confidence = float(probs.max())
            
            # Бинаризация маски
            mask_data = (probs > 0.5).astype(np.uint8) * 255

        # Шаг 4: Сохранение маски как отдельный PNG
        mask_name = f"{unique_id}_mask.png"
        cv2.imwrite(os.path.join(UPLOAD_DIR, mask_name), mask_data)

        # Шаг 5: Запись в базу данных
        new_rec = AnalysisRecord(
            filename=file.filename,
            original_path=f"/uploads/{orig_name}",
            mask_path=f"/uploads/{mask_name}"
        )
        db.add(new_rec)
        db.commit()

        print(f"--- [RESULT] Confidence: {confidence:.4f} ---")

        return {
            "status": "success",
            "confidence": round(confidence, 4),
            "original_image_url": f"http://127.0.0.1:8000/uploads/{orig_name}",
            "ai_mask_url": f"http://127.0.0.1:8000/uploads/{mask_name}"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def status():
    return {"message": "AI Medical System is Ready", "model": "Loaded (3ch)"}