# Fast application image — built from the pre-built local base image.
# Before using this Dockerfile you must build the base image once:
#   docker build -f backend/Dockerfile.base -t dhk-opencv-base:latest .
#
# After that, every docker-compose up --build only rebuilds this fast layer.
FROM dhk-opencv-base:latest

WORKDIR /app

# ---------- Python dependencies ----------
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# ---------- Application code ----------
# Keep the backend/ directory level so that config.py's
#   Path(__file__).resolve().parent.parent.parent
# resolves to /app (= PROJECT_ROOT), not to /.
COPY backend/app /app/backend/app

# Model source packages (needed for imports)
COPY world_model /app/world_model
COPY upscaler /app/upscaler
COPY interpolation /app/interpolation

ENV PYTHONUNBUFFERED=1
# /app/backend → makes `app.main` importable as a package
# /app         → makes `world_model`, `upscaler`, `interpolation` importable
ENV PYTHONPATH=/app/backend:/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
