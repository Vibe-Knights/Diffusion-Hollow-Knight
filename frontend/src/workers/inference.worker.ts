/// <reference lib="webworker" />

import * as ort from 'onnxruntime-web';
import { boxMullerRandom, buildSigmas, chwToRgba, encodeAction } from '../engine/sampler';
import { fetchWithCache } from '../engine/cache';
import type {
  EngineBackend,
  SeedManifestItem,
  WorkerInMessage,
  WorkerOutMessage,
} from '../types';

declare const self: DedicatedWorkerGlobalScope;

// Configure ONNX Runtime Web WASM binaries
ort.env.wasm.wasmPaths = '/ort-wasm/';
ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);

// Model and session state
let denoiserSession: ort.InferenceSession | null = null;
let upscalerSession: ort.InferenceSession | null = null;
let activeBackend: EngineBackend = 'webgpu';

// Dimensions
const CONTEXT_LEN = 4;
const H = 72;
const W = 128;
const LR_PLANE_SIZE = H * W;
const LR_CHW_SIZE = 3 * LR_PLANE_SIZE;
const OBS_BUFFER_SIZE = CONTEXT_LEN * LR_CHW_SIZE;

const HR_H = 288;
const HR_W = 512;
const HR_PLANE_SIZE = HR_H * HR_W;
const HR_RGBA_SIZE = HR_PLANE_SIZE * 4;
const LR_RGBA_SIZE = LR_PLANE_SIZE * 4;

// Buffers
const obsBuffer = new Float32Array(OBS_BUFFER_SIZE);
const actBuffer = new BigInt64Array(CONTEXT_LEN);
const xBuffer = new Float32Array(LR_CHW_SIZE);
const prevLrBuffer = new Float32Array(LR_CHW_SIZE);

// Double-buffering for Transferable ArrayBuffers to avoid memory allocations
let lrRgbaBuffer = new Uint8ClampedArray(LR_RGBA_SIZE);
let hrRgbaBuffer = new Uint8ClampedArray(HR_RGBA_SIZE);

// Sigmas schedule (3 steps)
const sigmas = buildSigmas(3, 0.002, 5.0, 7);

// Input and loop state
const pressedKeys = new Set<string>();
let isRunning = false;
let upscalerEnabled = true;
let isStepInProgress = false;

// Seed states
let seedManifest: SeedManifestItem[] = [];
let currentSeedId = 'seed_1';
let currentSeedData: ArrayBuffer | null = null;

// FPS calculation
let frameCounter = 0;
let lastFpsTime = performance.now();
let currentFps = 0;

function post(msg: WorkerOutMessage, transfer?: Transferable[]) {
  if (transfer && transfer.length > 0) {
    self.postMessage(msg, transfer);
  } else {
    self.postMessage(msg);
  }
}

async function loadSeed(seedId: string): Promise<boolean> {
  const item = seedManifest.find((s) => s.id === seedId) || seedManifest[0];
  if (!item) return false;

  try {
    const res = await fetch(`/${item.file}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const bin = await res.arrayBuffer();
    currentSeedData = bin;
    currentSeedId = item.id;

    // Load into obsBuffer
    const floatView = new Float32Array(bin);
    obsBuffer.set(floatView.subarray(0, OBS_BUFFER_SIZE));
    actBuffer.fill(0n);
    prevLrBuffer.set(floatView.subarray(OBS_BUFFER_SIZE - LR_CHW_SIZE, OBS_BUFFER_SIZE));

    post({ type: 'seed-changed', seedId: item.id });
    return true;
  } catch (err: any) {
    console.error(`[Worker] Failed to load seed ${seedId}:`, err);
    return false;
  }
}

async function initEngine(forceBackend?: EngineBackend) {
  try {
    post({ type: 'init-progress', stage: 'Fetching seeds manifest', percent: 5 });
    try {
      const manifestRes = await fetch('/seeds/manifest.json');
      if (manifestRes.ok) {
        seedManifest = await manifestRes.json();
      }
    } catch (e) {
      console.warn('[Worker] Could not fetch seeds manifest:', e);
    }

    // 1. Fetch Denoiser weights with progress
    post({ type: 'init-progress', stage: 'Downloading World Model (58MB)', percent: 10 });
    const denoiserBuffer = await fetchWithCache('/models/denoiser.onnx', (loaded, total, percent) => {
      const mapped = 10 + Math.round((percent / 100) * 45); // 10% to 55%
      post({
        type: 'init-progress',
        stage: `Downloading World Model: ${Math.round(loaded / 1e6)}MB / ${Math.round(total / 1e6)}MB`,
        percent: mapped,
      });
    });

    // 2. Fetch Upscaler weights
    post({ type: 'init-progress', stage: 'Downloading Neural Upscaler (6MB)', percent: 55 });
    const upscalerBuffer = await fetchWithCache('/models/upscaler.onnx', (loaded, total, percent) => {
      const mapped = 55 + Math.round((percent / 100) * 15); // 55% to 70%
      post({
        type: 'init-progress',
        stage: `Downloading Upscaler: ${Math.round(loaded / 1e6)}MB / ${Math.round(total / 1e6)}MB`,
        percent: mapped,
      });
    });

    // 3. Initialize Denoiser ONNX session (Try WebGPU first unless forced to WASM)
    if (forceBackend === 'wasm') {
      post({ type: 'init-progress', stage: 'Initializing WASM SIMD session', percent: 75 });
      denoiserSession = await ort.InferenceSession.create(denoiserBuffer, {
        executionProviders: ['wasm'],
      });
      activeBackend = 'wasm';
      console.log('[Worker] Denoiser initialized on WASM CPU (forced)!');
    } else {
      post({ type: 'init-progress', stage: 'Initializing WebGPU acceleration', percent: 75 });
      try {
        console.log('[Worker] Attempting WebGPU execution provider...');
        denoiserSession = await ort.InferenceSession.create(denoiserBuffer, {
          executionProviders: ['webgpu'],
        });
        activeBackend = 'webgpu';
        console.log('[Worker] Denoiser initialized on WebGPU!');
      } catch (gpuErr) {
        console.warn('[Worker] WebGPU failed, falling back to WASM SIMD:', gpuErr);
        post({ type: 'init-progress', stage: 'WebGPU unavailable, falling back to WASM SIMD', percent: 80 });
        denoiserSession = await ort.InferenceSession.create(denoiserBuffer, {
          executionProviders: ['wasm'],
        });
        activeBackend = 'wasm';
        console.log('[Worker] Denoiser initialized on WASM CPU!');
      }
    }

    // 4. Initialize Upscaler ONNX session
    post({ type: 'init-progress', stage: 'Initializing Upscaler session', percent: 88 });
    try {
      upscalerSession = await ort.InferenceSession.create(upscalerBuffer, {
        executionProviders: [activeBackend, 'wasm'],
      });
    } catch (e) {
      console.warn('[Worker] Upscaler initialization failed, upscaler disabled:', e);
      upscalerSession = null;
    }

    // 5. Load initial seed
    post({ type: 'init-progress', stage: 'Loading initial game state', percent: 95 });
    await loadSeed('seed_1');

    post({ type: 'init-progress', stage: 'Ready', percent: 100 });
    post({
      type: 'ready',
      backend: activeBackend,
      availableSeeds: seedManifest,
      activeSeed: currentSeedId,
    });
  } catch (err: any) {
    console.error('[Worker] Engine initialization error:', err);
    post({ type: 'error', message: err.message || 'Engine initialization failed' });
  }
}

async function step() {
  if (!denoiserSession || isStepInProgress) return;
  isStepInProgress = true;

  const tStart = performance.now();

  try {
    // 1. Update actions
    const currentAction = BigInt(encodeAction(pressedKeys));
    actBuffer[0] = actBuffer[1];
    actBuffer[1] = actBuffer[2];
    actBuffer[2] = actBuffer[3];
    actBuffer[3] = currentAction;

    // 2. Generate Gaussian noise x ~ N(0, 1)
    boxMullerRandom(xBuffer, 0, 1.0);

    // 3. EDM Euler sampling loop (3 steps)
    for (let s = 0; s < 3; s++) {
      const sigmaCurr = sigmas[s];
      const sigmaNext = sigmas[s + 1];

      const feeds = {
        noisy_next_obs: new ort.Tensor('float32', xBuffer, [1, 3, H, W]),
        sigma: new ort.Tensor('float32', Float32Array.from([sigmaCurr]), [1]),
        prev_obs: new ort.Tensor('float32', obsBuffer, [1, CONTEXT_LEN * 3, H, W]),
        prev_act: new ort.Tensor('int64', actBuffer, [1, CONTEXT_LEN]),
      };

      const results = await denoiserSession.run(feeds);
      const denoised = results.denoised.data as Float32Array;

      // Euler update: x = x + ((x - denoised) / sigmaCurr) * (sigmaNext - sigmaCurr)
      const dt = sigmaNext - sigmaCurr;
      const invSigma = 1.0 / sigmaCurr;
      for (let j = 0; j < LR_CHW_SIZE; j++) {
        const d = (xBuffer[j] - denoised[j]) * invSigma;
        xBuffer[j] += d * dt;
      }
    }

    // 4. Clamp generated frame to [0, 1]
    for (let j = 0; j < LR_CHW_SIZE; j++) {
      const val = xBuffer[j];
      xBuffer[j] = val < 0.0 ? 0.0 : val > 1.0 ? 1.0 : val;
    }

    // 5. Update context observation buffer
    obsBuffer.copyWithin(0, LR_CHW_SIZE);
    obsBuffer.set(xBuffer, (CONTEXT_LEN - 1) * LR_CHW_SIZE);

    // 6. Upscale or render directly
    let outWidth = W;
    let outHeight = H;
    let transferBuffer: ArrayBuffer;

    if (upscalerEnabled && upscalerSession) {
      outWidth = HR_W;
      outHeight = HR_H;

      const upscalerFeeds = {
        lr_current: new ort.Tensor('float32', xBuffer, [1, 3, H, W]),
        prev_lr: new ort.Tensor('float32', prevLrBuffer, [1, 3, H, W]),
      };

      const upResults = await upscalerSession.run(upscalerFeeds);
      const hrFrame = upResults.hr_frame.data as Float32Array;

      // Update previous low-res for next upscaler step
      prevLrBuffer.set(xBuffer);

      // Reallocate if detached by transfer
      if (hrRgbaBuffer.byteLength === 0) {
        hrRgbaBuffer = new Uint8ClampedArray(HR_RGBA_SIZE);
      }
      chwToRgba(hrFrame, hrRgbaBuffer, HR_H, HR_W);
      transferBuffer = hrRgbaBuffer.buffer;
      // create new reference for next frame
      hrRgbaBuffer = new Uint8ClampedArray(HR_RGBA_SIZE);
    } else {
      outWidth = W;
      outHeight = H;

      if (lrRgbaBuffer.byteLength === 0) {
        lrRgbaBuffer = new Uint8ClampedArray(LR_RGBA_SIZE);
      }
      chwToRgba(xBuffer, lrRgbaBuffer, H, W);
      transferBuffer = lrRgbaBuffer.buffer;
      lrRgbaBuffer = new Uint8ClampedArray(LR_RGBA_SIZE);
    }

    // Calculate FPS
    frameCounter++;
    const now = performance.now();
    const elapsed = now - lastFpsTime;
    if (elapsed >= 1000) {
      currentFps = Math.round((frameCounter * 1000) / elapsed);
      frameCounter = 0;
      lastFpsTime = now;
    }

    const stepTimeMs = now - tStart;

    // Send frame to main thread
    post(
      {
        type: 'frame',
        buffer: transferBuffer,
        width: outWidth,
        height: outHeight,
        fps: currentFps,
        stepTimeMs,
      },
      [transferBuffer]
    );
  } catch (err: any) {
    console.error('[Worker] Error during step:', err);
  } finally {
    isStepInProgress = false;
  }
}

async function loop() {
  if (!isRunning) return;
  await step();
  // Keep loop running without blocking worker message queue
  if (isRunning) {
    setTimeout(loop, 0);
  }
}

self.onmessage = async (e: MessageEvent<WorkerInMessage>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init':
      await initEngine(msg.forceBackend);
      break;

    case 'start':
      if (!isRunning) {
        isRunning = true;
        loop();
      }
      break;

    case 'stop':
      isRunning = false;
      break;

    case 'key':
      if (msg.state === 'down') {
        pressedKeys.add(msg.key.toLowerCase());
      } else {
        pressedKeys.delete(msg.key.toLowerCase());
      }
      break;

    case 'set-upscaler':
      upscalerEnabled = msg.enabled;
      break;

    case 'set-seed':
      await loadSeed(msg.seedId);
      break;

    case 'purge-buffer':
      if (currentSeedData) {
        const floatView = new Float32Array(currentSeedData);
        obsBuffer.set(floatView.subarray(0, OBS_BUFFER_SIZE));
        actBuffer.fill(0n);
        prevLrBuffer.set(floatView.subarray(OBS_BUFFER_SIZE - LR_CHW_SIZE, OBS_BUFFER_SIZE));
      }
      break;
  }
};
