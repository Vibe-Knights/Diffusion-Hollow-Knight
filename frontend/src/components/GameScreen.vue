<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import type { ConnectionState, EngineBackend, InputMessage } from '@/types'

const props = defineProps<{
  connectionState: ConnectionState
  error: string | null
  gameLoading: boolean
  loadingStage?: string
  loadingPercent?: number
  engineBackend?: EngineBackend
  fps?: number
  stepTimeMs?: number
}>()

const emit = defineEmits<{
  'keyaction': [msg: InputMessage]
  'frame': []
  'first-frame': []
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let ctx: CanvasRenderingContext2D | null = null
let currentImageData: ImageData | null = null
let firstFrameEmitted = false

function renderFrame(buffer: ArrayBuffer, width: number, height: number) {
  const canvas = canvasRef.value
  if (!canvas) return

  if (!ctx) {
    ctx = canvas.getContext('2d', { alpha: false, desynchronized: true })
    if (!ctx) return
  }

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
    currentImageData = ctx.createImageData(width, height)
  }

  if (currentImageData) {
    currentImageData.data.set(new Uint8ClampedArray(buffer))
    ctx.putImageData(currentImageData, 0, 0)
  }

  if (!firstFrameEmitted) {
    firstFrameEmitted = true
    emit('first-frame')
  }
  emit('frame')
}

defineExpose({
  renderFrame,
})

const activeKeys = ref<Record<string, boolean>>({})

const GAME_KEYS = new Set([
  'arrowleft', 'arrowright', 'arrowup', 'arrowdown',
  'a', 'd', 'w', 's',
  ' ', 'k', 'j',
])

function onKeyDown(e: KeyboardEvent) {
  const key = e.key.toLowerCase()
  if (GAME_KEYS.has(key)) {
    e.preventDefault()
    activeKeys.value[key] = true
    emit('keyaction', { key: e.key, state: 'down' })
  }
}

function onKeyUp(e: KeyboardEvent) {
  const key = e.key.toLowerCase()
  if (GAME_KEYS.has(key)) {
    e.preventDefault()
    activeKeys.value[key] = false
    emit('keyaction', { key: e.key, state: 'up' })
  }
}

onMounted(() => {
  window.addEventListener('keydown', onKeyDown)
  window.addEventListener('keyup', onKeyUp)
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDown)
  window.removeEventListener('keyup', onKeyUp)
})

function onDoubleClick() {
  if (canvasRef.value) {
    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      canvasRef.value.requestFullscreen()
    }
  }
}
</script>

<template>
  <main class="flex-1 flex items-center justify-center bg-hollow-bg relative overflow-hidden">
    <canvas
      ref="canvasRef"
      class="w-full h-full object-contain rounded shadow-2xl"
      style="image-rendering: pixelated;"
      @dblclick="onDoubleClick"
    />

    <!-- Real-time HUD (FPS, Latency & Engine) overlay -->
    <div
      v-if="connectionState === 'connected'"
      class="absolute top-4 right-4 bg-black/70 backdrop-blur-md border border-white/10 rounded-lg px-3 py-1.5 flex items-center gap-3 text-xs font-mono pointer-events-none z-20 shadow-2xl"
    >
      <div class="flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full" :class="(fps ?? 0) >= 15 ? 'bg-emerald-400' : (fps ?? 0) >= 8 ? 'bg-amber-400' : 'bg-cyan-400 animate-pulse'"></span>
        <span class="text-white font-bold">{{ (fps ?? 0) > 0 ? (fps ?? 0).toFixed(1) : '--' }}</span>
        <span class="text-white/60 text-[10px]">FPS</span>
      </div>
      <div class="h-3 w-[1px] bg-white/20"></div>
      <div class="flex items-center gap-1 text-white/80">
        <span>{{ (stepTimeMs ?? 0) > 0 ? stepTimeMs : '--' }}</span>
        <span class="text-white/60 text-[10px]">ms</span>
      </div>
      <div class="h-3 w-[1px] bg-white/20"></div>
      <span class="text-[10px] uppercase tracking-wider text-hollow-accent font-semibold">
        {{ engineBackend || 'WEBGPU' }}
      </span>
    </div>

    <div class="absolute bottom-6 left-6 p-4 rounded-lg flex gap-8 z-20 items-end">
      <div class="flex flex-col items-center gap-2">
        <div class="flex justify-center w-full">
          <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['w'] ? 'bg-white/20 scale-95 text-white' : 'bg-black/40 text-white/70']">
            <span class="text-[16px] font-bold">W</span>
            <span class="text-[9px] opacity-70 mt-0.5">Jump</span>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['a'] ? 'bg-white/20 scale-95 text-white' : 'bg-black/40 text-white/70']">
            <span class="text-[16px] font-bold">A</span>
            <span class="text-[9px] opacity-70 mt-0.5">Left</span>
          </div>
          <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['s'] ? 'bg-white/20 scale-95 text-white' : 'bg-black/40 text-white/70']">
            <span class="text-[16px] font-bold">S</span>
            <span class="text-[9px] opacity-70 mt-0.5">Down</span>
          </div>
          <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['d'] ? 'bg-white/20 scale-95 text-white' : 'bg-black/40 text-white/70']">
            <span class="text-[16px] font-bold">D</span>
            <span class="text-[9px] opacity-70 mt-0.5">Right</span>
          </div>
        </div>
      </div>
      
      <div class="flex items-center gap-2">
        <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['k'] ? 'bg-white/20 scale-95 text-white' : 'bg-black/40 text-white/70']">
          <span class="text-[16px] font-bold">K</span>
          <span class="text-[9px] opacity-70 mt-0.5">Attack</span>
        </div>
        <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['j'] ? 'bg-white/20 scale-95 text-white' : 'bg-black/40 text-white/70']">
          <span class="text-[16px] font-bold">J</span>
          <span class="text-[9px] opacity-70 mt-0.5">Heal</span>
        </div>
      </div>
    </div>

    <!-- Idle state overlay -->
    <div
      v-if="connectionState === 'idle'"
      class="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-hollow-bg/80"
    >
      <p class="text-3xl font-bold text-hollow-accent tracking-wider glow-text">Diffusion Hollow Knight</p>
      <p class="text-hollow-muted text-sm">Client-Side World Model (WebGPU + onnxruntime-web)</p>
      <p class="text-hollow-muted text-xs mt-2">Press <span class="font-mono text-hollow-text font-bold bg-white/10 px-2 py-1 rounded">START GAME</span> to begin</p>
    </div>

    <!-- Connecting / Loading overlay -->
    <div
      v-if="gameLoading"
      class="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-hollow-bg/85 backdrop-blur-sm"
    >
      <svg class="animate-spin h-8 w-8 text-hollow-accent" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
      </svg>
      <p class="text-hollow-accent font-bold">{{ loadingStage || 'Warming up diffusion model…' }}</p>
      
      <div v-if="loadingPercent !== undefined && loadingPercent > 0" class="w-64 bg-white/10 rounded-full h-2 overflow-hidden mt-2">
        <div
          class="bg-hollow-accent h-full transition-all duration-150 ease-out rounded-full"
          :style="{ width: `${loadingPercent}%` }"
        ></div>
      </div>
      <p v-if="loadingPercent !== undefined && loadingPercent > 0" class="text-hollow-muted text-xs font-mono">
        {{ loadingPercent }}%
      </p>
    </div>

    <!-- Error state overlay -->
    <div
      v-if="connectionState === 'error'"
      class="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-hollow-bg/80"
    >
      <p class="text-red-400 font-bold">Engine Error</p>
      <p class="text-hollow-muted text-sm max-w-md text-center">{{ error }}</p>
    </div>

    <!-- Disconnected state overlay -->
    <div
      v-if="connectionState === 'disconnected'"
      class="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-hollow-bg/80"
    >
      <p class="text-yellow-400 font-bold">Game Paused</p>
      <p class="text-hollow-muted text-sm">Press START GAME to resume</p>
    </div>
  </main>
</template>
