<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import ControlPanel from './components/ControlPanel.vue'
import GameScreen from './components/GameScreen.vue'
import StatusBar from './components/StatusBar.vue'
import { useWebRTC } from './composables/useWebRTC'
import type { ServerConfig } from './types'

const {
  connectionState,
  remoteStream,
  fps,
  error,
  connect,
  disconnect,
  sendInput,
  sendSettings,
  countFrame,
} = useWebRTC()

const serverConfig = ref<ServerConfig | null>(null)
const upscalerOn = ref(true)
const interpolationOn = ref(false)
const interpolationExp = ref(1)
const opticalFlowOn = ref(false)
const flowBackend = ref('NVOF')
const upscaleModel = ref('')
const interpolationModel = ref('')
const gameLoading = ref(false)

let configPollTimer: ReturnType<typeof setInterval> | null = null

async function fetchConfig() {
  try {
    const res = await fetch('/api/config')
    if (res.ok) {
      const cfg: ServerConfig = await res.json()
      serverConfig.value = cfg
      if (!configPollTimer) {
        upscalerOn.value = cfg.upscaler_available
        interpolationOn.value = false
        interpolationExp.value = cfg.interpolation_exp
        opticalFlowOn.value = cfg.use_optical_flow
        
        flowBackend.value = cfg.flow_backends?.[0] || 'NVOF'
        upscaleModel.value = cfg.upscale_models?.[0] || ''
        interpolationModel.value = cfg.interpolation_models?.[0] || ''
      }
    }
  } catch {
    console.warn('Could not fetch server config')
  }
}

function startConfigPolling() {
  stopConfigPolling()
  configPollTimer = setInterval(fetchConfig, 3000)
}

function stopConfigPolling() {
  if (configPollTimer) {
    clearInterval(configPollTimer)
    configPollTimer = null
  }
}

function onToggleUpscaler(val: boolean) {
  upscalerOn.value = val
  sendSettings({ type: 'settings', upscaler: val })
}

function onToggleInterpolation(val: boolean) {
  interpolationOn.value = val
  sendSettings({ type: 'settings', interpolation: val })
}

function onChangeExp(val: number) {
  interpolationExp.value = val
  sendSettings({ type: 'settings', interpolation_exp: val })
}

function onToggleOpticalFlow(val: boolean) {
  opticalFlowOn.value = val
  sendSettings({ type: 'settings', use_optical_flow: val })
}

function onFlowBackendChange(val: string) {
  flowBackend.value = val
  sendSettings({ type: 'settings', flow_backend: val })
}

function onUpscaleModelChange(val: string) {
  upscaleModel.value = val
  sendSettings({ type: 'settings', upscale_model: val })
}

function onInterpolationModelChange(val: string) {
  interpolationModel.value = val
  sendSettings({ type: 'settings', interpolation_model: val })
}

async function onConnect() {
  gameLoading.value = true
  await connect()
}

async function onDisconnect() {
  gameLoading.value = false
  await disconnect()
}

function onFirstFrame() {
  gameLoading.value = false
}

watch(connectionState, (state) => {
  if (state === 'disconnected' || state === 'error' || state === 'idle') {
    gameLoading.value = false
  }
})

onMounted(() => {
  fetchConfig()
  startConfigPolling()
})

onUnmounted(() => {
  stopConfigPolling()
})
</script>

<template>
  <div class="flex flex-col h-full min-h-screen font-body-lg dark bg-[#0e0e0e] text-on-surface">
    <ControlPanel
      :connected="connectionState === 'connected'"
      :upscaler-on="upscalerOn"
      :interpolation-on="interpolationOn"
      :upscaler-available="serverConfig?.upscaler_available ?? false"
      :interpolation-available="serverConfig?.interpolation_available ?? false"
      :interpolation-models="serverConfig?.interpolation_models ?? []"
      :upscale-models="serverConfig?.upscale_models ?? []"
      :flow-backends="serverConfig?.flow_backends ?? []"
      :interpolation-exp="interpolationExp"
      :optical-flow-on="opticalFlowOn"
      :flow-backend="flowBackend"
      :upscale-model="upscaleModel"
      :interpolation-model="interpolationModel"
      :nvof-available="serverConfig?.nvof_available ?? false"
      @toggle-upscaler="onToggleUpscaler"
      @toggle-interpolation="onToggleInterpolation"
      @change-exp="onChangeExp"
      @toggle-optical-flow="onToggleOpticalFlow"
      @change-flow-backend="onFlowBackendChange"
      @change-upscale-model="onUpscaleModelChange"
      @change-interpolation-model="onInterpolationModelChange"
      @connect="onConnect"
      @disconnect="onDisconnect"
    />

    <div class="flex flex-1 pt-16 h-screen w-full">
      <main class="flex-1 ml-0 md:ml-64 relative overflow-hidden p-4 h-full">
        <GameScreen
          :stream="remoteStream"
          :connection-state="connectionState"
          :error="error"
          :game-loading="gameLoading"
          @keyaction="sendInput"
          @frame="countFrame"
          @first-frame="onFirstFrame"
        />
      </main>
    </div>

    <StatusBar
      class="hidden"
      :connection-state="connectionState"
      :fps="fps"
      :active-sessions="serverConfig?.active_sessions ?? 0"
      :max-sessions="serverConfig?.max_sessions ?? 0"
    />
  </div>
</template>
