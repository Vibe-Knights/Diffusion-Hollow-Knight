<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import type { ConnectionState, InputMessage } from '@/types'

const props = defineProps<{
  stream: MediaStream | null
  connectionState: ConnectionState
  error: string | null
  gameLoading: boolean
}>()

const emit = defineEmits<{
  'keyaction': [msg: InputMessage]
  'frame': []
  'first-frame': []
}>()

const videoRef = ref<HTMLVideoElement | null>(null)

// Bind stream to video element; reset first-frame flag on new stream
watch(() => props.stream, (stream) => {
  firstFrameEmitted = false
  rvfcActive = false
  if (videoRef.value && stream) {
    videoRef.value.srcObject = stream
  }
})

let rvfcActive = false

function onVideoFrame() {
  emit('frame')
  if (videoRef.value && !videoRef.value.paused) {
    videoRef.value.requestVideoFrameCallback(onVideoFrame)
  } else {
    rvfcActive = false
  }
}

function tryStartFrameCounter() {
  if (rvfcActive) return
  const v = videoRef.value
  if (!v || v.paused) return
  try {
    v.requestVideoFrameCallback(onVideoFrame)
    rvfcActive = true
  } catch {
  }
}

let firstFrameEmitted = false

function onVideoPlay() {
  tryStartFrameCounter()
  if (!firstFrameEmitted) {
    firstFrameEmitted = true
    emit('first-frame')
  }
}

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
  rvfcActive = false
})

function onDoubleClick() {
  if (videoRef.value) {
    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      videoRef.value.requestFullscreen()
    }
  }
}
</script>

<template>
  <main class="flex-1 flex items-center justify-center bg-hollow-bg relative overflow-hidden">
    <video
      ref="videoRef"
      autoplay
      playsinline
      muted
      class="w-full h-full object-contain rounded shadow-2xl"
      style="image-rendering: pixelated;"
      @play="onVideoPlay"
      @dblclick="onDoubleClick"
    />

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
      
      <div class="flex items-center">
        <div :class="['w-12 h-12 flex flex-col items-center justify-center font-label-caps rounded-xl border border-white/20 transition-all duration-75', activeKeys['k'] ? 'bg-white/20 scale-95 text-white' : 'bg-black/40 text-white/70']">
          <span class="text-[16px] font-bold">K</span>
          <span class="text-[9px] opacity-70 mt-0.5">Attack</span>
        </div>
      </div>
    </div>

    <div
      v-if="connectionState === 'idle'"
      class="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-hollow-bg/80"
    >
      <p class="text-2xl font-bold text-hollow-accent">Diffusion Hollow Knight</p>
      <p class="text-hollow-muted">Press <span class="font-mono text-hollow-text">Connect</span> to start</p>
    </div>

    <div
      v-if="connectionState === 'connecting'"
      class="absolute inset-0 flex items-center justify-center bg-hollow-bg/80"
    >
      <div class="flex items-center gap-3">
        <svg class="animate-spin h-6 w-6 text-hollow-accent" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
        </svg>
        <span class="text-hollow-text">Connecting…</span>
      </div>
    </div>

    <div
      v-if="connectionState === 'connected' && gameLoading"
      class="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-hollow-bg/80"
    >
      <svg class="animate-spin h-8 w-8 text-hollow-accent" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
      </svg>
      <p class="text-hollow-accent font-bold">Loading game…</p>
      <p class="text-hollow-muted text-xs">Warming up diffusion model, this may take a minute</p>
    </div>

    <div
      v-if="connectionState === 'error'"
      class="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-hollow-bg/80"
    >
      <p class="text-red-400 font-bold">Connection Error</p>
      <p class="text-hollow-muted text-sm">{{ error }}</p>
    </div>

    <div
      v-if="connectionState === 'disconnected'"
      class="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-hollow-bg/80"
    >
      <p class="text-yellow-400 font-bold">Disconnected</p>
      <p class="text-hollow-muted text-sm">Press Connect to rejoin</p>
    </div>
  </main>
</template>
