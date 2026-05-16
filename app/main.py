import os
import datetime
import uuid
import torch
import torch.nn as nn
import torchvision.transforms as T
from datetime import date
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pydicom
import numpy as np
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Импортируем твою модель (убедись, что файл app/ml/unet_model.py заполнен кодом)
from .ml.unet_model import UNet 

# --- НАСТРОЙКИ БД ---
DATABASE_URL = "sqlite:///./medical_data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnalysisRecord(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String)
    filename = Column(String)      # Уникальное имя DICOM
    preview_path = Column(String)  # Уникальное имя PNG
    prediction = Column(String)
    confidence = Column(Float)
    tumor_size = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- ИНИЦИАЛИЗАЦИЯ ИИ МОДЕЛИ ---
device = torch.device("cpu")

# ВАЖНО: n_channels=3, так как твои веса ожидают RGB вход
model = UNet(n_channels=3, n_classes=1) 
model.load_state_dict(torch.load("app/ml/best_model.pth", map_location=device))
model.eval()

# Подготовка снимка под требования модели (Resize 256x256)
preprocess = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# --- ПРИЛОЖЕНИЕ ---
app = FastAPI(title="Система ИИ-диагностики рака яичников")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# --- ЭНДПОИНТЫ ---

@app.post("/upload-dicom", tags=["Анализ снимков"])
async def upload_dicom(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Создаем уникальное имя
        unique_prefix = str(uuid.uuid4())[:8]
        unique_filename = f"{unique_prefix}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Чтение DICOM
        ds = pydicom.dcmread(file_path)
        patient_id = f"ANON_{ds.get('PatientID', 'Unknown')}"
        spacing = [float(s) for s in ds.get("PixelSpacing", [1.0, 1.0])]

        # --- РАБОТА ИИ ---
        pixel_array = ds.pixel_array
        # Нормализация и перевод в 3 канала (RGB), так как этого требуют веса .pth
        pixel_array_norm = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array) + 1e-7)
        pixel_array_uint8 = (pixel_array_norm * 255).astype(np.uint8)
        
        img_for_ai = Image.fromarray(pixel_array_uint8).convert("RGB")
        input_tensor = preprocess(img_for_ai).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            # Получаем маску уверенности
            prob_mask = torch.sigmoid(output).squeeze().numpy()
            mask = prob_mask > 0.5 # Порог 50%

        # Поиск границ опухоли на маске
        coords = np.argwhere(mask)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            
            # Масштабируем координаты обратно под оригинал
            orig_h, orig_w = pixel_array.shape
            x0, x1 = int(x0 * orig_w / 256), int(x1 * orig_w / 256)
            y0, y1 = int(y0 * orig_h / 256), int(y1 * orig_h / 256)

            prediction = "Обнаружено новообразование"
            confidence = float(round(np.max(prob_mask) * 100, 2))
            tumor_size_text = f"{round((x1-x0)*spacing[1], 1)} x {round((y1-y0)*spacing[0], 1)} мм"
            is_tumor = True
        else:
            prediction = "Патологий не выявлено"
            confidence = 0.0
            tumor_size_text = "—"
            is_tumor = False

        # --- ВИЗУАЛИЗАЦИЯ (Отрисовка маски) ---
        rescaled = (np.maximum(pixel_array, 0) / np.max(pixel_array) * 255).astype(np.uint8)
        final_image = Image.fromarray(rescaled).convert("RGB")
        
        if is_tumor:
            draw = ImageDraw.Draw(final_image)
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, y0-15), f"AI: {confidence}%", fill="red")

        preview_filename = f"preview_{unique_filename}.png"
        preview_path = os.path.join(UPLOAD_DIR, preview_filename)
        final_image.save(preview_path)

        # Сохранение в БД
        new_record = AnalysisRecord(
            patient_id=patient_id, filename=unique_filename,
            preview_path=preview_path, prediction=prediction, 
            confidence=confidence, tumor_size=tumor_size_text
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        return {"status": "success", "data": new_record}
    except Exception as e:
        return {"status": "error", "message": f"Ошибка анализа: {str(e)}"}

@app.get("/history", tags=["История пациентов"])
def get_history(
    target_date: date = Query(None, description="Дата ГГГГ-ММ-ДД"),
    only_today: bool = Query(False, description="Только за сегодня"),
    db: Session = Depends(get_db)
):
    query = db.query(AnalysisRecord)
    if only_today:
        today = datetime.datetime.utcnow().date()
        query = query.filter(func.date(AnalysisRecord.timestamp) == today)
    elif target_date:
        query = query.filter(func.date(AnalysisRecord.timestamp) == target_date)
    return query.order_by(AnalysisRecord.timestamp.desc()).all()

@app.get("/history/{patient_id}", tags=["История пациентов"])
def get_patient_history(patient_id: str, db: Session = Depends(get_db)):
    results = db.query(AnalysisRecord).filter(AnalysisRecord.patient_id.contains(patient_id)).all()
    if not results:
        raise HTTPException(status_code=404, detail="Пациент не найден")
    return results

@app.delete("/delete/{record_id}", tags=["Управление данными"])
def delete_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Запись не найдена")

    if os.path.exists(record.preview_path): os.remove(record.preview_path)
    original_dicom = os.path.join(UPLOAD_DIR, record.filename)
    if os.path.exists(original_dicom): os.remove(original_dicom)

    db.delete(record)
    db.commit()
    return {"status": "success", "message": f"Запись {record_id} удалена"}