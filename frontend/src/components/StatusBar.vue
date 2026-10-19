<script setup lang="ts">
import type { ConnectionState, EngineBackend } from '@/types'

defineProps<{
  connectionState: ConnectionState
  fps?: number
  stepTimeMs?: number
  engineBackend?: EngineBackend
  activeSessions?: number
  maxSessions?: number
}>()

const stateLabels: Record<ConnectionState, string> = {
  idle: 'Engine Idle',
  connecting: 'Initializing WebGPU…',
  connected: 'Engine Running',
  disconnected: 'Paused',
  error: 'Engine Error',
}

const stateColors: Record<ConnectionState, string> = {
  idle: 'bg-outline-variant',
  connecting: 'bg-amber-400',
  connected: 'bg-emerald-400',
  disconnected: 'bg-rose-400',
  error: 'bg-red-500',
}
</script>

<template>
  <footer class="h-9 ml-0 md:ml-64 px-6 bg-surface-container-lowest border-t border-outline-variant/10 flex items-center justify-between text-xs text-on-surface-variant shrink-0 z-30 font-mono">
    <div class="flex items-center gap-5">
      <!-- Connection indicator -->
      <div class="flex items-center gap-2">
        <span class="w-2 h-2 rounded-full" :class="stateColors[connectionState]" />
        <span class="text-on-surface font-sans font-medium text-[11px]">{{ stateLabels[connectionState] }}</span>
      </div>

      <!-- FPS & Latency -->
      <div v-if="connectionState === 'connected'" class="flex items-center gap-3">
        <span class="text-on-surface font-bold">
          {{ (fps ?? 0) > 0 ? (fps ?? 0).toFixed(1) : '--' }} FPS
        </span>
        <span v-if="(stepTimeMs ?? 0) > 0" class="text-on-surface-variant/70 text-[11px]">
          {{ stepTimeMs }} ms
        </span>
        <span class="px-1.5 py-0.2 rounded bg-surface-container-high text-[10px] text-primary uppercase font-bold">
          {{ engineBackend || 'WEBGPU' }}
        </span>
      </div>
    </div>

    <div class="flex items-center gap-5">
      <!-- Client Architecture Badge -->
      <span class="hidden sm:inline text-on-surface-variant/60 text-[11px]">
        DIAMOND World Model • In-Browser Client Inference
      </span>

      <!-- Controls hint -->
      <span class="text-on-surface-variant/80 text-[11px] font-sans">
        WASD — move &nbsp; Space / W — jump &nbsp; K — attack &nbsp; J — heal
      </span>
    </div>
  </footer>
</template>
