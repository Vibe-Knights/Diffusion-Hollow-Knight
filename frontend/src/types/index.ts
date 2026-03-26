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
}

export type ConnectionState = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error'
