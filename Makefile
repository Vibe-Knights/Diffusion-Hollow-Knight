# ── Base images (run ONCE, pick one) ─────────────────────────────
# Full: OpenCV CUDA + cupy + NVOF  (~30 min build)
build-base:
	docker build -f backend/Dockerfile.base -t dhk-opencv-base:latest .

# Lite: pip opencv-python-headless, no NVOF/cupy  (~2 min build)
build-base-lite:
	docker build -f backend/Dockerfile.base.lite -t dhk-opencv-base:latest .

# ── App ──────────────────────────────────────────────────────────
up:
	docker compose up --build

build:
	docker compose build

down:
	docker compose down

.PHONY: build-base build-base-lite up build down
