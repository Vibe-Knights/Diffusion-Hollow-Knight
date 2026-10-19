import { ref, type Ref, shallowRef } from 'vue';
import type {
  ConnectionState,
  EngineBackend,
  InputMessage,
  SeedManifestItem,
  SettingsMessage,
  WorkerInMessage,
  WorkerOutMessage,
} from '@/types';

// Import Web Worker using Vite's ?worker constructor
import InferenceWorker from '@/workers/inference.worker?worker';

export function useGameEngine() {
  const connectionState: Ref<ConnectionState> = ref('idle');
  const engineBackend: Ref<EngineBackend> = ref('webgpu');
  const fps: Ref<number> = ref(0);
  const stepTimeMs: Ref<number> = ref(0);
  const error: Ref<string | null> = ref(null);
  const gameLoading: Ref<boolean> = ref(false);
  const loadingStage: Ref<string> = ref('');
  const loadingPercent: Ref<number> = ref(0);

  const upscalerOn: Ref<boolean> = ref(false);
  const upscalerAvailable: Ref<boolean> = ref(true);
  const samplingSteps: Ref<number> = ref(2);

  const availableSeeds: Ref<SeedManifestItem[]> = ref([]);
  const activeSeed: Ref<string> = ref('seed_1');

  let worker: Worker | null = null;
  let hasReceivedFirstFrame = false;

  type FrameCallback = (buffer: ArrayBuffer, width: number, height: number) => void;
  const frameCallbacks: FrameCallback[] = [];

  function onFrame(cb: FrameCallback) {
    frameCallbacks.push(cb);
  }

  function initWorker(): Worker | null {
    if (worker) return worker;

    try {
      worker = new InferenceWorker();
    } catch (e: any) {
      console.error('[GameEngine] Worker instantiation failed:', e);
      error.value = 'Failed to create worker: ' + e.message;
      connectionState.value = 'error';
      return null;
    }

    worker.onmessage = (e: MessageEvent<WorkerOutMessage>) => {
      const msg = e.data;
      switch (msg.type) {
        case 'init-progress':
          loadingStage.value = msg.stage;
          loadingPercent.value = msg.percent;
          break;

        case 'ready':
          engineBackend.value = msg.backend;
          availableSeeds.value = msg.availableSeeds;
          activeSeed.value = msg.activeSeed;
          gameLoading.value = false;
          connectionState.value = 'connected';
          // Auto-start generation upon ready
          worker?.postMessage({ type: 'start' });
          break;

        case 'frame':
          fps.value = msg.fps;
          stepTimeMs.value = Math.round(msg.stepTimeMs);
          if (!hasReceivedFirstFrame) {
            hasReceivedFirstFrame = true;
            gameLoading.value = false;
          }
          for (const cb of frameCallbacks) {
            cb(msg.buffer, msg.width, msg.height);
          }
          break;

        case 'seed-changed':
          activeSeed.value = msg.seedId;
          break;

        case 'error':
          console.error('[GameEngine] Worker error message:', msg.message);
          error.value = msg.message;
          connectionState.value = 'error';
          gameLoading.value = false;
          break;
      }
    };

    worker.onerror = (err) => {
      console.error('[GameEngine] Worker runtime error:', err);
      error.value = err.message || 'Worker runtime failure';
      connectionState.value = 'error';
      gameLoading.value = false;
    };

    return worker;
  }

  async function connect(forceBackend?: EngineBackend): Promise<void> {
    error.value = null;
    gameLoading.value = true;
    loadingStage.value = 'Initializing engine…';
    loadingPercent.value = 0;
    hasReceivedFirstFrame = false;
    connectionState.value = 'connecting';

    const w = initWorker();
    if (w) {
      w.postMessage({ type: 'init', baseUrl: window.location.origin, forceBackend });
      w.postMessage({ type: 'start' });
    }
  }

  async function disconnect(): Promise<void> {
    if (worker) {
      worker.postMessage({ type: 'stop' });
    }
    connectionState.value = 'disconnected';
    gameLoading.value = false;
    fps.value = 0;
  }

  function sendInput(msg: InputMessage): void {
    if (worker && connectionState.value === 'connected') {
      worker.postMessage({ type: 'key', key: msg.key, state: msg.state });
    }
  }

  function sendSettings(settings: SettingsMessage): void {
    if (!worker) return;

    if (settings.upscaler !== undefined) {
      upscalerOn.value = settings.upscaler;
      worker.postMessage({ type: 'set-upscaler', enabled: settings.upscaler });
    }

    if (settings.steps !== undefined) {
      samplingSteps.value = settings.steps;
      worker.postMessage({ type: 'set-steps', steps: settings.steps });
    }

    if (settings.seed_id !== undefined) {
      activeSeed.value = settings.seed_id;
      worker.postMessage({ type: 'set-seed', seedId: settings.seed_id });
    }
  }

  function purgeBuffer(): void {
    if (worker) {
      worker.postMessage({ type: 'purge-buffer' });
    }
  }

  function setSeed(seedId: string): void {
    if (worker) {
      worker.postMessage({ type: 'set-seed', seedId });
    }
  }

  return {
    connectionState,
    engineBackend,
    fps,
    stepTimeMs,
    error,
    gameLoading,
    loadingStage,
    loadingPercent,
    upscalerOn,
    upscalerAvailable,
    samplingSteps,
    availableSeeds,
    activeSeed,
    connect,
    disconnect,
    sendInput,
    sendSettings,
    purgeBuffer,
    setSeed,
    onFrame,
  };
}
