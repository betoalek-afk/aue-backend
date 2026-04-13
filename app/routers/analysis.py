from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
import os
import numpy as np
from ..database import get_db
from ..models import AnalysisRecord
from ..services.dicom_logic import process_dicom

router = APIRouter(prefix="/analysis", tags=["Medical Analysis"])

UPLOAD_DIR = "uploads"

# 1. Загрузка и анализ
@router.post("/upload")
async def upload_dicom(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    p_id, spacing, preview = process_dicom(file_path)

    prediction = "Опухоль обнаружена" if np.random.random() > 0.5 else "Опухоль не обнаружена"
    confidence = float(round(np.random.uniform(70, 99), 2))

    new_record = AnalysisRecord(
        patient_id=f"ANON_{p_id}",
        filename=file.filename,
        prediction=prediction,
        confidence=confidence
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)

    return {"status": "success", "analysis": new_record}

# 2. Вся история
@router.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(AnalysisRecord).order_by(AnalysisRecord.timestamp.desc()).all()

# 3. Поиск конкретного пациента (Блок ИСТОРИЯ ПАЦИЕНТОВ)
@router.get("/history/{patient_id}")
def get_patient_history(patient_id: str, db: Session = Depends(get_db)):
    results = db.query(AnalysisRecord).filter(AnalysisRecord.patient_id == patient_id).all()
    if not results:
        raise HTTPException(status_code=404, detail="Пациент не найден")
    return results

# 4. УДАЛЕНИЕ (с очисткой папки uploads)
@router.delete("/delete/{record_id}")
def delete_record(record_id: int, db: Session = Depends(get_db)):
    # Ищем запись в базе
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Запись не найдена")

    # Удаляем физические файлы с диска (Блок ПОСТ-ПРОЦЕССОР / Очистка)
    file_path = os.path.join(UPLOAD_DIR, record.filename)
    if os.path.exists(file_path):
        os.remove(file_path) # удаляем .dcm
    if os.path.exists(f"{file_path}.png"):
        os.remove(f"{file_path}.png") # удаляем .png

    # Удаляем из базы
    db.delete(record)
    db.commit()

    return {"status": "success", "message": f"Запись {record_id} и связанные файлы удалены"}