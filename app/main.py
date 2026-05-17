import os
import sys
import uuid
import datetime
from datetime import timezone
import torch
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

# Настройка путей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 1. ОЧИСТКА МАСКИ ---
def soft_clean(probs, threshold=0.3):
    # Бинаризация
    mask = (probs > threshold).astype(np.uint8) * 255
    if mask.max() == 0:
        return mask
    # Убираем мелкий шум морфологией
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# --- 2. БАЗА ДАННЫХ ---
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

# --- 3. ПРИЛОЖЕНИЕ ---
app = FastAPI(title="AI Ovarian Segmentation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# --- 4. ЗАГРУЗКА МОДЕЛИ ---
from app.ml.unet_model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=1).to(device)
model_path = os.path.join(os.path.dirname(__file__), "ml", "best_model.pth")

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ [AI-SYSTEM] Модель загружена на {device}!")
    except Exception as e:
        print(f"❌ [ERROR] Ошибка загрузки весов: {e}")
else:
    print(f"⚠️ [WARNING] Файл весов не найден по пути: {model_path}")

# --- 5. ЭНДПОИНТ ---
@app.post("/upload-dicom")
async def upload_dicom(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        unique_id = str(uuid.uuid4())
        dicom_path = os.path.join(UPLOAD_DIR, f"{unique_id}.dcm")
        
        with open(dicom_path, "wb") as b:
            b.write(await file.read())

        # 1. Чтение DICOM
        ds = pydicom.dcmread(dicom_path)
        img_array = apply_voi_lut(ds.pixel_array, ds)
        
        # 2. Нормализация для визуализации (0-255)
        img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-5) * 255.0
        img_uint8 = np.uint8(img_norm)

        # 3. Подготовка для модели (RGB + Resize 256x256)
        if len(img_uint8.shape) == 2:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
            
        img_res = cv2.resize(img_rgb, (256, 256))
        
        # Сохраняем оригинал для фронтенда
        orig_name = f"{unique_id}_original.jpg"
        cv2.imwrite(os.path.join(UPLOAD_DIR, orig_name), cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR))

        # 4. ИНФЕРЕНС (Самое важное!)
        # Порядок: [H,W,C] -> [C,H,W], деление на 255.0, добавление Batch-размерности
        input_tensor = torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        with torch.no_grad():
            output = model(input_tensor.to(device))
            probs = torch.sigmoid(output).squeeze().cpu().numpy()
            max_conf = float(probs.max())
            
            # Получаем чистую маску
            mask_data = soft_clean(probs, threshold=0.3)
            
            # Если маска пустая, пробуем порог пониже
            if mask_data.max() == 0:
                mask_data = (probs > 0.15).astype(np.uint8) * 255

        # 5. Сохранение маски
        mask_name = f"{unique_id}_mask.png"
        cv2.imwrite(os.path.join(UPLOAD_DIR, mask_name), mask_data)

        # 6. Запись в базу
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
            "original_image_url": f"/uploads/{orig_name}",
            "ai_mask_url": f"/uploads/{mask_name}"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health(): return {"status": "online"}