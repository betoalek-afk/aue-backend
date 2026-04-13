import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import numpy as np
import datetime
import os

def create_dummy_dicom(filename="test.dcm"):
    # 1. Метаданные (техническая часть для pydicom)
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian 

    # 2. Создаем сам файл данных
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Заполняем "Паспорт" пациента (блок АНОНИМИЗАТОР его потом изменит)
    ds.PatientName = "CITIZEN^Joe"
    ds.PatientID = "123456"
    
    # Дата и время "съемки"
    now = datetime.datetime.now()
    ds.ContentDate = now.strftime('%Y%m%d')
    ds.ContentTime = now.strftime('%H%M%S.%f')
    
    # ВАЖНО: Масштаб (сколько мм в одном пикселе)
    # Это нужно для блока "ПАНЕЛЬ АНАЛИЗА", чтобы считать размер опухоли
    ds.PixelSpacing = [0.5, 0.5] 

    # 3. Настройки изображения
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0 

    # 4. Генерируем "шум" (имитация УЗИ-снимка)
    # Создаем серый фон с небольшими случайными точками
    pixel_data = np.random.randint(100, 500, (512, 512), dtype=np.uint16)
    ds.PixelData = pixel_data.tobytes()

    # Сохраняем
    ds.save_as(filename)
    print(f"✅ Тестовый файл '{filename}' успешно создан!")
    print(f"Путь: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_dummy_dicom()