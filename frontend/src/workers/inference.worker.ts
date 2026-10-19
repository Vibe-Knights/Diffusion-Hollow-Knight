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

// Configurable sampling steps (default 1 for 20 FPS ultra performance, 2 for balanced, 3 for quality)
let samplingSteps = 1;
let sigmas = buildSigmas(samplingSteps, 0.002, 5.0, 7);
let sigmaTensors: ort.Tensor[] = [];

function rebuildSigmas(steps: number) {
  samplingSteps = Math.max(1, Math.min(steps, 10));
  sigmas = buildSigmas(samplingSteps, 0.002, 5.0, 7);
  sigmaTensors = [];
  for (let s = 0; s < samplingSteps; s++) {
    sigmaTensors.push(new ort.Tensor('float32', Float32Array.from([sigmas[s]]), [1]));
  }
}
rebuildSigmas(samplingSteps);

// Input and loop state
const pressedKeys = new Set<string>();
let isRunning = false;
let upscalerEnabled = false; // Default to false (72p pixel-art) for blazing performance, user can toggle 288p on GPU
let isStepInProgress = false;

// Seed states
let seedManifest: SeedManifestItem[] = [];
let currentSeedId = 'seed_1';
let currentSeedData: ArrayBuffer | null = null;

// FPS calculation with rolling average excluding initial warmup frames
let currentFps = 0;
let frameCount = 0;
const recentStepTimes: number[] = [];

// High-performance macrotask yield using MessageChannel (0.05ms vs 4.2ms setTimeout)
const yieldChannel = new MessageChannel();
let yieldResolve: (() => void) | null = null;
yieldChannel.port1.onmessage = () => {
  if (yieldResolve) {
    const r = yieldResolve;
    yieldResolve = null;
    r();
  }
};
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    yieldResolve = resolve;
    yieldChannel.port2.postMessage(null);
  });
}

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
    const webgpuOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: [
        {
          name: 'webgpu',
          preferredLayout: 'NCHW',
        },
      ],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
    };

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
        denoiserSession = await ort.InferenceSession.create(denoiserBuffer, webgpuOptions);
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

    // 4. Initialize Upscaler ONNX session on WebGPU
    post({ type: 'init-progress', stage: 'Initializing FastUpscaler (GPU)', percent: 88 });
    try {
      if (activeBackend === 'webgpu') {
        console.log('[Worker] Initializing FastUpscaler on WebGPU...');
        upscalerSession = await ort.InferenceSession.create(upscalerBuffer, webgpuOptions);
        console.log('[Worker] FastUpscaler successfully running on WebGPU!');
      } else {
        upscalerSession = await ort.InferenceSession.create(upscalerBuffer, {
          executionProviders: ['wasm'],
        });
        console.log('[Worker] FastUpscaler running on WASM CPU!');
      }
    } catch (e) {
      console.warn('[Worker] WebGPU for Upscaler failed, falling back to WASM:', e);
      try {
        upscalerSession = await ort.InferenceSession.create(upscalerBuffer, {
          executionProviders: ['wasm'],
        });
        console.log('[Worker] FastUpscaler fallback to WASM CPU succeeded!');
      } catch (err2) {
        console.error('[Worker] Upscaler initialization failed completely:', err2);
        upscalerSession = null;
      }
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

    // Pre-allocate context tensors once per frame to eliminate GC and buffer creation overhead
    const prevObsTensor = new ort.Tensor('float32', obsBuffer, [1, CONTEXT_LEN * 3, H, W]);
    const prevActTensor = new ort.Tensor('int64', actBuffer, [1, CONTEXT_LEN]);

    // 3. EDM Euler sampling loop
    if (samplingSteps === 1) {
      // 1-step direct denoise: with sigmaCurr = 5.0 and sigmaNext = 0.0,
      // the Euler update formula yields x = denoised directly.
      const feeds = {
        noisy_next_obs: new ort.Tensor('float32', xBuffer, [1, 3, H, W]),
        sigma: sigmaTensors[0],
        prev_obs: prevObsTensor,
        prev_act: prevActTensor,
      };

      const results = await denoiserSession.run(feeds);
      const denoised = results.denoised.data as Float32Array;
      xBuffer.set(denoised);
    } else {
      for (let s = 0; s < samplingSteps; s++) {
        const sigmaCurr = sigmas[s];
        const sigmaNext = sigmas[s + 1];

        const feeds = {
          noisy_next_obs: new ort.Tensor('float32', xBuffer, [1, 3, H, W]),
          sigma: sigmaTensors[s],
          prev_obs: prevObsTensor,
          prev_act: prevActTensor,
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

    const tDenoise = performance.now() - tStart;
    let tUpscale = 0;

    if (upscalerEnabled && upscalerSession) {
      outWidth = HR_W;
      outHeight = HR_H;

      const tUpStart = performance.now();
      const upscalerFeeds = {
        lr_current: new ort.Tensor('float32', xBuffer, [1, 3, H, W]),
        prev_lr: new ort.Tensor('float32', prevLrBuffer, [1, 3, H, W]),
      };

      const upResults = await upscalerSession.run(upscalerFeeds);
      const hrFrame = upResults.hr_frame.data as Float32Array;
      tUpscale = performance.now() - tUpStart;

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

    // Calculate FPS with rolling average, skipping the first warmup frame
    const now = performance.now();
    const stepTimeMs = now - tStart;
    frameCount++;

    if (frameCount > 1) {
      recentStepTimes.push(stepTimeMs);
      if (recentStepTimes.length > 6) {
        recentStepTimes.shift();
      }
      const avgStepTime = recentStepTimes.reduce((acc, v) => acc + v, 0) / recentStepTimes.length;
      currentFps = avgStepTime > 0 ? (1000 / avgStepTime) : 0;
    } else {
      currentFps = stepTimeMs > 0 ? (1000 / stepTimeMs) : 0;
    }

    console.log(`[Worker Frame] Total: ${stepTimeMs.toFixed(1)}ms (Denoiser ${samplingSteps}x: ${tDenoise.toFixed(1)}ms, Upscaler: ${tUpscale.toFixed(1)}ms) | FPS: ${currentFps.toFixed(1)}`);

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
  while (isRunning) {
    await step();
    await yieldToEventLoop();
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

    case 'set-steps':
      rebuildSigmas(msg.steps);
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
