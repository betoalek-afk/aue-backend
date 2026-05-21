# AUE Backend: Medical Image Analysis (U-Net & FastAPI) 🩺

Бэкенд-система для автоматизированного анализа медицинских изображений в формате DICOM. Проект объединяет современный веб-фреймворк FastAPI и нейросетевую модель глубокого обучения для сегментации медицинских снимков.

---

## 🌟 Основные возможности
* **DICOM Processing:** Чтение, обработка и извлечение метаданных из медицинских файлов (`.dcm`).
* **AI Segmentation:** Использование нейронной сети архитектуры **U-Net** (PyTorch) для анализа снимков.
* **Image Serving:** Конвертация DICOM в PNG для отображения в веб-интерфейсе.
* **Data Management:** Хранение результатов анализа и истории в базе данных SQLite.
* **Production Ready:** Запуск через производительный сервер `Waitress`.

---

## 🛠 Технологический стек
- **API:** FastAPI, Pydantic, a2wsgi
- **Нейросеть:** PyTorch (torch, torchvision)
- **Обработка изображений:** OpenCV (cv2), PyDicom, Pillow (PIL)
- **База данных:** SQLAlchemy (SQLite)
- **Сервер:** Waitress

---

## 📂 Структура проекта
- `app/ml/` — Архитектура модели U-Net и веса `best_model.pth`.
- `app/routers/` — Маршруты API (загрузка, поиск по истории).
- `app/services/` — Обработка DICOM и инференс нейросети.
- `app/models.py` — Схемы таблиц базы данных.
- `test/` — Скрипты для генерации тестовых данных.

---

## ⚙️ Установка и запуск

### 1. Подготовка окружения
```bash
python -m venv .venv
# Для Windows:
.venv\Scripts\activate
# Для Linux/macOS:
source .venv/bin/activate 
```
### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```
### 3. Запуск сервера
```bash
python run_server.py
```

## 📝 API Endpoints
* **POST** `/analysis/upload` — Загрузка DICOM и получение результата.
* **GET** `/analysis/history` — Список всех проведенных анализов.
* **GET** `/analysis/history/{patient_id}` — Поиск результатов по ID пациента.