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

const activeKeys = ref<Record<string, boolean>>({
  w: false,
  a: false,
  s: false,
  d: false,
  k: false,
})

// Action mapping supporting EN/RU layouts, arrow keys, and standard platformer bindings
function resolveActionKey(e: KeyboardEvent): string | null {
  const code = e.code
  const key = e.key ? e.key.toLowerCase() : ''

  // W / Jump (W, ArrowUp, Space, Cyrillic Ц)
  if (code === 'KeyW' || code === 'ArrowUp' || code === 'Space' || key === 'w' || key === 'arrowup' || key === ' ' || key === 'ц') {
    return 'w'
  }
  // A / Left (A, ArrowLeft, Cyrillic Ф)
  if (code === 'KeyA' || code === 'ArrowLeft' || key === 'a' || key === 'arrowleft' || key === 'ф') {
    return 'a'
  }
  // S / Down (S, ArrowDown, Cyrillic Ы)
  if (code === 'KeyS' || code === 'ArrowDown' || key === 's' || key === 'arrowdown' || key === 'ы') {
    return 's'
  }
  // D / Right (D, ArrowRight, Cyrillic В)
  if (code === 'KeyD' || code === 'ArrowRight' || key === 'd' || key === 'arrowright' || key === 'в') {
    return 'd'
  }
  // K / Attack (K, KeyZ, KeyX, Cyrillic Л, Я, Ч)
  if (code === 'KeyK' || code === 'KeyZ' || code === 'KeyX' || key === 'k' || key === 'z' || key === 'x' || key === 'л') {
    return 'k'
  }
  return null
}

function onKeyDown(e: KeyboardEvent) {
  const actionKey = resolveActionKey(e)
  if (actionKey) {
    e.preventDefault()
    activeKeys.value[actionKey] = true
    emit('keyaction', { key: actionKey, state: 'down' })
  }
}

function onKeyUp(e: KeyboardEvent) {
  const actionKey = resolveActionKey(e)
  if (actionKey) {
    e.preventDefault()
    activeKeys.value[actionKey] = false
    emit('keyaction', { key: actionKey, state: 'up' })
  }
}

function onWindowBlur() {
  for (const k of Object.keys(activeKeys.value)) {
    if (activeKeys.value[k]) {
      activeKeys.value[k] = false
      emit('keyaction', { key: k, state: 'up' })
    }
  }
}

onMounted(() => {
  window.addEventListener('keydown', onKeyDown)
  window.addEventListener('keyup', onKeyUp)
  window.addEventListener('blur', onWindowBlur)
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDown)
  window.removeEventListener('keyup', onKeyUp)
  window.removeEventListener('blur', onWindowBlur)
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

    <!-- Original In-Game Control Key Indicator (W, A, S, D, K) -->
    <div class="absolute bottom-6 left-6 p-4 rounded-lg flex gap-8 z-20 items-end pointer-events-none">
      <div class="flex flex-col items-center gap-2">
        <div class="flex justify-center w-full">
          <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['w'] ? 'bg-white/25 scale-95 text-white ring-2 ring-white/50 shadow-lg' : 'bg-black/40 text-white/70']">
            <span class="text-[16px] font-bold">W</span>
            <span class="text-[9px] opacity-70 mt-0.5">Jump</span>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['a'] ? 'bg-white/25 scale-95 text-white ring-2 ring-white/50 shadow-lg' : 'bg-black/40 text-white/70']">
            <span class="text-[16px] font-bold">A</span>
            <span class="text-[9px] opacity-70 mt-0.5">Left</span>
          </div>
          <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['s'] ? 'bg-white/25 scale-95 text-white ring-2 ring-white/50 shadow-lg' : 'bg-black/40 text-white/70']">
            <span class="text-[16px] font-bold">S</span>
            <span class="text-[9px] opacity-70 mt-0.5">Down</span>
          </div>
          <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['d'] ? 'bg-white/25 scale-95 text-white ring-2 ring-white/50 shadow-lg' : 'bg-black/40 text-white/70']">
            <span class="text-[16px] font-bold">D</span>
            <span class="text-[9px] opacity-70 mt-0.5">Right</span>
          </div>
        </div>
      </div>
      
      <div class="flex items-center">
        <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['k'] ? 'bg-white/25 scale-95 text-white ring-2 ring-white/50 shadow-lg' : 'bg-black/40 text-white/70']">
          <span class="text-[16px] font-bold">K</span>
          <span class="text-[9px] opacity-70 mt-0.5">Attack</span>
        </div>
      </div>
    </div>

    <!-- Idle state overlay -->
    <div
      v-if="connectionState === 'idle'"
      class="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-hollow-bg/80"
    >
      <p class="text-2xl font-bold text-hollow-accent">Diffusion Hollow Knight</p>
      <p class="text-hollow-muted">Press <span class="font-mono text-hollow-text">START STREAM</span> to start</p>
    </div>

    <!-- Connecting / Loading overlay -->
    <div
      v-if="connectionState === 'connecting' || gameLoading"
      class="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-hollow-bg/80"
    >
      <svg class="animate-spin h-8 w-8 text-hollow-accent" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
      </svg>
      <p class="text-hollow-accent font-bold">{{ loadingStage || 'Connecting…' }}</p>
      <div v-if="loadingPercent !== undefined && loadingPercent > 0" class="w-64 bg-white/10 rounded-full h-2 overflow-hidden mt-2">
        <div
          class="bg-hollow-accent h-full transition-all duration-150 ease-out rounded-full"
          :style="{ width: `${loadingPercent}%` }"
        ></div>
      </div>
    </div>

    <!-- Error state overlay -->
    <div
      v-if="connectionState === 'error'"
      class="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-hollow-bg/80"
    >
      <p class="text-red-400 font-bold">Connection Error</p>
      <p class="text-hollow-muted text-sm max-w-md text-center">{{ error }}</p>
    </div>

    <!-- Disconnected state overlay -->
    <div
      v-if="connectionState === 'disconnected'"
      class="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-hollow-bg/80"
    >
      <p class="text-yellow-400 font-bold">Disconnected</p>
      <p class="text-hollow-muted text-sm">Press START STREAM to rejoin</p>
    </div>
  </main>
</template>
