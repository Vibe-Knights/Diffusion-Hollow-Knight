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

# ── Client-side Web (Static Frontend + WebGPU/WASM) ─────────────
export-models:
	python3 scripts/export_all.py

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-preview:
	cd frontend && npm run preview

.PHONY: build-base build-base-lite up build down export-models frontend-install frontend-dev frontend-build frontend-preview

