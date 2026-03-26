from __future__ import annotations
import torch
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.pipeline.diffusion import load_diffusion
from app.pipeline.upscaler_loader import load_upscaler, is_nvof_available
from app.pipeline.interpolator_loader import load_interpolator
from app.signaling import router as signaling_router, init_shared_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading AI models …")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True 

    sampler = load_diffusion(settings)
    upscaler = load_upscaler(settings)
    interpolator = load_interpolator(settings)

    init_shared_models(sampler, upscaler, interpolator, nvof_available=is_nvof_available())
    log.info("All models loaded. Server ready.")

    yield

    log.info("Shutting down …")


app = FastAPI(title="Diffusion Hollow Knight", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signaling_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
