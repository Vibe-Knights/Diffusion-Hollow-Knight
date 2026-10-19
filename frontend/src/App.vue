<script setup lang="ts">
import { onMounted, ref } from 'vue'
import ControlPanel from './components/ControlPanel.vue'
import GameScreen from './components/GameScreen.vue'
import StatusBar from './components/StatusBar.vue'
import { useGameEngine } from './composables/useGameEngine'

const {
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
  clearCacheAndReload,
  onFrame,
} = useGameEngine()

const gameScreenRef = ref<InstanceType<typeof GameScreen> | null>(null)

const opticalFlowOn = ref(false)
const flowBackend = ref('Zero')
const upscaleModel = ref('FastUpscaler 1.6M')
const interpolationOn = ref(false)
const interpolationExp = ref(1)
const interpolationModel = ref('v1.0')

function onToggleUpscaler(val: boolean) {
  sendSettings({ type: 'settings', upscaler: val })
}

function onChangeSteps(val: number) {
  sendSettings({ type: 'settings', steps: val })
}

function onToggleInterpolation(val: boolean) {
  interpolationOn.value = val
}

function onChangeExp(val: number) {
  interpolationExp.value = val
}

function onToggleOpticalFlow(val: boolean) {
  opticalFlowOn.value = val
}

function onFlowBackendChange(val: string) {
  flowBackend.value = val
}

function onUpscaleModelChange(val: string) {
  upscaleModel.value = val
}

function onInterpolationModelChange(val: string) {
  interpolationModel.value = val
}

function onChangeSeed(seedId: string) {
  setSeed(seedId)
}

function onPurgeBuffer() {
  purgeBuffer()
}

function onConnect() {
  const urlParams = new URLSearchParams(window.location.search)
  const forced = urlParams.get('backend') === 'wasm' ? 'wasm' : undefined
  connect(forced)
}

onMounted(() => {
  onFrame((buffer, width, height) => {
    gameScreenRef.value?.renderFrame(buffer, width, height)
  })
})
</script>

<template>
  <div class="flex flex-col h-screen overflow-hidden font-body-lg dark bg-[#0e0e0e] text-on-surface">
    <ControlPanel
      :connected="connectionState === 'connected'"
      :upscaler-on="upscalerOn"
      :interpolation-on="interpolationOn"
      :upscaler-available="upscalerAvailable"
      :interpolation-available="false"
      :interpolation-models="['v1.0']"
      :upscale-models="['FastUpscaler 1.6M']"
      :flow-backends="['Zero', 'NVOF', 'OpenCV', 'IFNet']"
      :interpolation-exp="interpolationExp"
      :optical-flow-on="opticalFlowOn"
      :flow-backend="flowBackend"
      :upscale-model="upscaleModel"
      :interpolation-model="interpolationModel"
      :nvof-available="false"
      :engine-backend="engineBackend"
      :available-seeds="availableSeeds"
      :active-seed="activeSeed"
      :fps="fps"
      :step-time-ms="stepTimeMs"
      :sampling-steps="samplingSteps"
      @toggle-upscaler="onToggleUpscaler"
      @change-steps="onChangeSteps"
      @toggle-interpolation="onToggleInterpolation"
      @change-exp="onChangeExp"
      @toggle-optical-flow="onToggleOpticalFlow"
      @change-flow-backend="onFlowBackendChange"
      @change-upscale-model="onUpscaleModelChange"
      @change-interpolation-model="onInterpolationModelChange"
      @change-seed="onChangeSeed"
      @purge-buffer="onPurgeBuffer"
      @clear-cache="clearCacheAndReload"
      @connect="onConnect"
      @disconnect="disconnect"
    />

    <div class="flex flex-1 pt-16 overflow-hidden w-full">
      <main class="flex-1 ml-0 md:ml-64 relative overflow-hidden p-4 h-full">
        <GameScreen
          ref="gameScreenRef"
          :connection-state="connectionState"
          :error="error"
          :game-loading="gameLoading"
          :loading-stage="loadingStage"
          :loading-percent="loadingPercent"
          :engine-backend="engineBackend"
          :fps="fps"
          :step-time-ms="stepTimeMs"
          @keyaction="sendInput"
        />
      </main>
    </div>

    <StatusBar
      :connection-state="connectionState"
      :fps="fps"
      :step-time-ms="stepTimeMs"
      :engine-backend="engineBackend"
      :active-sessions="connectionState === 'connected' ? 1 : 0"
      :max-sessions="1"
    />
  </div>
</template>
