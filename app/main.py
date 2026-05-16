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

# Импортируем архитектуру UNet из твоей папки app/ml
from .ml.unet_model import UNet 

# --- НАСТРОЙКИ БАЗЫ ДАННЫХ ---
DATABASE_URL = "sqlite:///./medical_data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnalysisRecord(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String)
    filename = Column(String)      
    preview_path = Column(String)  
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ВАЖНО: n_channels=3, так как веса .pth ожидают RGB
model = UNet(n_channels=3, n_classes=1) 
model.load_state_dict(torch.load("app/ml/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Подготовка снимка под требования модели (Resize 256x256)
preprocess = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# --- ПРИЛОЖЕНИЕ ---
app = FastAPI(title="Система ИИ-диагностики рака яичников")

# Разрешаем фронтенду (Лере) подключаться
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
        # Уникальное имя для файлов
        unique_prefix = str(uuid.uuid4())[:8]
        unique_filename = f"{unique_prefix}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Чтение DICOM
        ds = pydicom.dcmread(file_path)
        patient_id = f"ANON_{ds.get('PatientID', 'Unknown')}"
        spacing = [float(s) for s in ds.get("PixelSpacing", [1.0, 1.0])]

        # --- ПОДГОТОВКА ИЗОБРАЖЕНИЯ (Защита от многомерных массивов) ---
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Решение ошибки "too many values to unpack": берем последние 2 измерения (H, W)
        orig_h, orig_w = pixel_array.shape[-2:]
        
        # Схлопываем лишние измерения (например, если снимок 1x512x512)
        if pixel_array.ndim > 2:
            pixel_array = np.squeeze(pixel_array)
            if pixel_array.ndim > 2: # Если это серия кадров, берем первый
                pixel_array = pixel_array[0]

        # Нормализация
        p_min, p_max = np.min(pixel_array), np.max(pixel_array)
        if p_max - p_min == 0: p_max += 1e-7
        pixel_array_uint8 = (255.0 * (pixel_array - p_min) / (p_max - p_min)).astype(np.uint8)
        
        # Перевод в RGB (3 канала) для нейросети
        img_for_ai = Image.fromarray(pixel_array_uint8).convert("RGB")
        input_tensor = preprocess(img_for_ai).unsqueeze(0).to(device)

        # --- ИНФЕРЕНС (РАБОТА ИИ) ---
        with torch.no_grad():
            output = model(input_tensor)
            # Приводим маску к 2D виду
            prob_mask = torch.sigmoid(output).cpu().numpy()
            while prob_mask.ndim > 2:
                prob_mask = prob_mask[0]

        max_prob = float(np.max(prob_mask))
        confidence = round(max_prob * 100, 2)
        
        # Порог срабатывания 0.4
        threshold = 0.4 
        mask = prob_mask > threshold
        coords = np.argwhere(mask)
        is_tumor = coords.size > 0

        # --- ВИЗУАЛИЗАЦИЯ И РЕЗАЙЗ 256x256 ---
        final_image = img_for_ai.resize((256, 256))
        
        if is_tumor:
            # Координаты на маске 256x256
            y0, x0 = coords.min(axis=0)[-2:]
            y1, x1 = coords.max(axis=0)[-2:]
            
            # Расчет физического размера в мм с учетом оригинального разрешения
            tumor_size_text = f"{round((x1-x0)*(orig_w/256)*spacing[1], 1)} x {round((y1-y0)*(orig_h/256)*spacing[0], 1)} мм"
            prediction = "Обнаружено новообразование"
            
            # Рисуем рамку на превью 256x256
            draw = ImageDraw.Draw(final_image)
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, max(0, y0-15)), f"AI: {confidence}%", fill="red")
        else:
            prediction = "Патологий не выявлено"
            tumor_size_text = "—"

        # Сохранение превью 256x256
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
        import traceback
        print(traceback.format_exc()) # Логируем ошибку в терминал
        return {"status": "error", "message": f"Ошибка анализа: {str(e)}"}

@app.get("/history", tags=["История пациентов"])
def get_history(
    target_date: date = Query(None, description="ГГГГ-ММ-ДД"),
    only_today: bool = Query(False, description="За сегодня"),
    db: Session = Depends(get_db)
):
    query = db.query(AnalysisRecord)
    if only_today:
        query = query.filter(func.date(AnalysisRecord.timestamp) == datetime.datetime.utcnow().date())
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
    original_path = os.path.join(UPLOAD_DIR, record.filename)
    if os.path.exists(original_path): os.remove(original_path)

    db.delete(record)
    db.commit()
    return {"status": "success"}

@app.get("/", tags=["Главная"])
def welcome():
    return {"message": "AI Medical Backend 256px Version is Ready"}