import pydicom
import numpy as np
import os
from PIL import Image

def process_dicom(file_path: str):
    ds = pydicom.dcmread(file_path)
    
    # Извлекаем данные
    raw_spacing = ds.get("PixelSpacing", [0.5, 0.5])
    spacing = [float(s) for s in raw_spacing] if hasattr(raw_spacing, "__iter__") else [0.5, 0.5]
    patient_id = str(ds.get("PatientID", "Unknown"))

    # Конвертация в PNG
    pixel_array = ds.pixel_array
    max_val = np.max(pixel_array)
    rescaled = (np.maximum(pixel_array, 0) / max_val * 255) if max_val > 0 else pixel_array
    img = Image.fromarray(rescaled.astype(np.uint8))
    
    preview_path = f"{file_path}.png"
    img.save(preview_path)
    
    return patient_id, spacing, preview_path