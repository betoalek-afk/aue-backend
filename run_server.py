from waitress import serve
from a2wsgi import ASGIMiddleware
from app.main import app

# Этот "мостик" позволяет Waitress работать с FastAPI
wsgi_app = ASGIMiddleware(app)

if __name__ == "__main__":
    print("--- Сервер запускается на http://127.0.0.1:8000 ---")
    print("--- Нажми Ctrl+C, чтобы остановить ---")
    serve(wsgi_app, host='127.0.0.1', port=8000, threads=4)