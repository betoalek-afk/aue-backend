import os
import uuid
import datetime
from datetime import timezone
import torch
import torch.nn as nn
from torchvision import transforms as T
import pydicom
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- БАЗА ДАННЫХ ---
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

# --- ПРИЛОЖЕНИЕ ---
app = FastAPI(title="AI Medical API - Side-by-Side Mode")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# --- МОДЕЛЬ ---
device = torch.device("cpu")
model = None

try:
    from app.ml.unet_model import UNet
    def load_model():
        global model
        model_path = "app/ml/best_model.pth"
        if os.path.exists(model_path):
            try:
                # Создаем модель (3 канала на вход, как требует твой .pth)
                try: model = UNet(n_channels=3, n_classes=1)
                except: model = UNet(in_channels=3, out_channels=1)
                
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                print("✅ [SUCCESS] ИИ-модель готова к работе!")
            except Exception as e:
                print(f"❌ [ERROR] Ошибка весов: {e}")
    load_model()
except Exception as e:
    print(f"❌ [ERROR] Ошибка импорта модели: {e}")

preprocess = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# --- ЭНДПОИНТ ---

@app.post("/upload-dicom")
async def upload_dicom(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        unique_id = str(uuid.uuid4())
        
        # 1. Сохраняем DICOM
        dicom_path = os.path.join(UPLOAD_DIR, f"{unique_id}.dcm")
        with open(dicom_path, "wb") as b: b.write(await file.read())

        # 2. Обработка изображения
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255.0
        
        # СОХРАНЯЕМ ОРИГИНАЛ КАК JPG
        orig_pill = Image.fromarray(np.uint8(img)).convert("RGB")
        orig_name = f"{unique_id}_original.jpg"
        orig_pill.save(os.path.join(UPLOAD_DIR, orig_name), format="JPEG", quality=95)

        mask_name = "no_mask_found.png"
        confidence = 0.0
        
        if model:
            # 3. Предсказание ИИ
            input_t = preprocess(orig_pill).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_t)
                if isinstance(output, (dict, list)): output = output[0]
                
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                confidence = float(probs.max())
                
                # Порог (0.5). Если ИИ уверен, создаем маску
                mask_data = (probs > 0.5).astype(np.uint8) * 255

            # 4. СОХРАНЯЕМ МАСКУ КАК ОТДЕЛЬНЫЙ PNG
            mask_pill = Image.fromarray(mask_data).resize(orig_pill.size, Image.NEAREST).convert("L")
            mask_name = f"{unique_id}_mask.png"
            mask_pill.save(os.path.join(UPLOAD_DIR, mask_name))
            
            print(f"--- [AI] Уверенность: {confidence:.4f}. Маска сохранена. ---")

        # 5. Запись в базу
        new_rec = AnalysisRecord(
            filename=file.filename,
            original_path=f"/uploads/{orig_name}",
            mask_path=f"/uploads/{mask_name}"
        )
        db.add(new_rec); db.commit()

        return {
            "status": "success",
            "confidence": round(confidence, 4),
            "original_image_url": f"http://127.0.0.1:8000/uploads/{orig_name}",
            "ai_mask_url": f"http://127.0.0.1:8000/uploads/{mask_name}"
        }

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))