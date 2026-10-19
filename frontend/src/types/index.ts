export interface ServerConfig {
  fps: number
  max_sessions: number
  active_sessions: number
  upscaler_available: boolean
  interpolation_available: boolean
  interpolation_models: string[]
  interpolation_exp: number
  nvof_available: boolean
  use_optical_flow: boolean
  flow_backends: string[]
  upscale_models: string[]
}

export interface InputMessage {
  key: string
  state: 'down' | 'up'
}

export interface SettingsMessage {
  type: 'settings'
  upscaler?: boolean
  interpolation?: boolean
  interpolation_exp?: number
  use_optical_flow?: boolean
  flow_backend?: string
  upscale_model?: string
  interpolation_model?: string
  seed_id?: string
}

export type ConnectionState = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error'

export type EngineBackend = 'webgpu' | 'wasm'

export interface SeedManifestItem {
  id: string
  name: string
  file: string
  thumbnail: string
  frames: number
  shape: number[]
}

export interface EngineStats {
  fps: number
  stepTimeMs: number
  backend: EngineBackend
  activeSeed: string
}

// Worker message protocols
export type WorkerInMessage =
  | { type: 'init'; baseUrl: string; forceBackend?: EngineBackend }
  | { type: 'start' }
  | { type: 'stop' }
  | { type: 'key'; key: string; state: 'down' | 'up' }
  | { type: 'set-upscaler'; enabled: boolean }
  | { type: 'set-seed'; seedId: string }
  | { type: 'purge-buffer' }

export type WorkerOutMessage =
  | { type: 'init-progress'; stage: string; percent: number; detail?: string }
  | { type: 'ready'; backend: EngineBackend; availableSeeds: SeedManifestItem[]; activeSeed: string }
  | { type: 'frame'; buffer: ArrayBuffer; width: number; height: number; fps: number; stepTimeMs: number }
  | { type: 'seed-changed'; seedId: string }
  | { type: 'error'; message: string }
