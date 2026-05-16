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

# Импорт модели
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

# ВАЖНО: Убедись, что n_channels совпадает с тем, на чем обучали (3 для RGB)
model = UNet(n_channels=3, n_classes=1) 
model.load_state_dict(torch.load("app/ml/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Препроцессинг должен быть ОДИН В ОДИН как при обучении
preprocess = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(), # Автоматически делит на 255
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

@app.post("/upload-dicom", tags=["Анализ снимков"])
async def upload_dicom(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        unique_prefix = str(uuid.uuid4())[:8]
        unique_filename = f"{unique_prefix}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Чтение DICOM
        ds = pydicom.dcmread(file_path)
        patient_id = f"ANON_{ds.get('PatientID', 'Unknown')}"
        spacing = [float(s) for s in ds.get("PixelSpacing", [1.0, 1.0])]

        # --- ПОДГОТОВКА ИЗОБРАЖЕНИЯ ---
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Улучшенная нормализация (стабильнее к шумам)
        p_min, p_max = np.min(pixel_array), np.max(pixel_array)
        if p_max - p_min == 0: p_max += 1e-7
        
        # Приводим к 0-255 для корректной работы PIL и преобразования в RGB
        pixel_array_uint8 = (255.0 * (pixel_array - p_min) / (p_max - p_min)).astype(np.uint8)
        
        img_for_ai = Image.fromarray(pixel_array_uint8).convert("RGB")
        input_tensor = preprocess(img_for_ai).unsqueeze(0).to(device)

        # --- ИНФЕРЕНС (ПРЕДСКАЗАНИЕ) ---
        with torch.no_grad():
            output = model(input_tensor)
            prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()

        # Считаем максимальную уверенность нейросети на всем снимке
        max_prob = float(np.max(prob_mask))
        confidence = round(max_prob * 100, 2)
        
        # Порог срабатывания (если больше 0.4 — считаем патологией)
        threshold = 0.4 
        mask = prob_mask > threshold

        # Логируем в консоль для отладки
        print(f"--- АНАЛИЗ ФАЙЛА {file.filename} ---")
        print(f"Max Probability: {max_prob:.4f} (Threshold: {threshold})")

        coords = np.argwhere(mask)
        is_tumor = coords.size > 0

        if is_tumor:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            
            # Масштабируем координаты рамки обратно под размер оригинала
            orig_h, orig_w = pixel_array.shape
            x0, x1 = int(x0 * orig_w / 256), int(x1 * orig_w / 256)
            y0, y1 = int(y0 * orig_h / 256), int(y1 * orig_h / 256)

            prediction = "Обнаружено новообразование"
            tumor_size_text = f"{round((x1-x0)*spacing[1], 1)} x {round((y1-y0)*spacing[0], 1)} мм"
        else:
            prediction = "Патологий не выявлено"
            tumor_size_text = "—"

        # --- ВИЗУАЛИЗАЦИЯ ---
        # Для превью используем ту же нормализованную картинку
        final_image = Image.fromarray(pixel_array_uint8).convert("RGB")
        
        if is_tumor:
            draw = ImageDraw.Draw(final_image)
            draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
            # Рисуем текст с уверенностью
            draw.text((x0, max(0, y0-20)), f"AI: {confidence}%", fill="red")

        preview_filename = f"preview_{unique_filename}.png"
        preview_path = os.path.join(UPLOAD_DIR, preview_filename)
        final_image.save(preview_path)

        # Сохранение в БД
        new_record = AnalysisRecord(
            patient_id=patient_id, 
            filename=unique_filename,
            preview_path=preview_path, 
            prediction=prediction, 
            confidence=confidence, # Теперь тут реальное число, а не всегда 0
            tumor_size=tumor_size_text
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        return {"status": "success", "data": new_record}
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"status": "error", "message": f"Ошибка анализа: {str(e)}"}

# Остальные эндпоинты (history, delete) остаются без изменений...
@app.get("/history", tags=["История пациентов"])
def get_history(target_date: date = Query(None), only_today: bool = Query(False), db: Session = Depends(get_db)):
    query = db.query(AnalysisRecord)
    if only_today:
        query = query.filter(func.date(AnalysisRecord.timestamp) == datetime.datetime.utcnow().date())
    elif target_date:
        query = query.filter(func.date(AnalysisRecord.timestamp) == target_date)
    return query.order_by(AnalysisRecord.timestamp.desc()).all()

@app.delete("/delete/{record_id}", tags=["Управление данными"])
def delete_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record: raise HTTPException(status_code=404)
    if os.path.exists(record.preview_path): os.remove(record.preview_path)
    os.remove(os.path.join(UPLOAD_DIR, record.filename))
    db.delete(record)
    db.commit()
    return {"status": "success"}