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
}

export type ConnectionState = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error'
