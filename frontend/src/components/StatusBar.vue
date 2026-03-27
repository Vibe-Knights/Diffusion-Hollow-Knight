<script setup lang="ts">
import type { ConnectionState } from '@/types'

defineProps<{
  connectionState: ConnectionState
  fps: number
  activeSessions: number
  maxSessions: number
}>()

const stateLabels: Record<ConnectionState, string> = {
  idle: 'Idle',
  connecting: 'Connecting…',
  connected: 'Connected',
  disconnected: 'Disconnected',
  error: 'Error',
}

const stateColors: Record<ConnectionState, string> = {
  idle: 'bg-hollow-muted',
  connecting: 'bg-yellow-400',
  connected: 'bg-green-400',
  disconnected: 'bg-red-400',
  error: 'bg-red-500',
}
</script>

<template>
  <footer class="flex items-center justify-between px-6 py-2 bg-hollow-panel border-t border-hollow-border text-xs text-hollow-muted shrink-0">
    <div class="flex items-center gap-4">
      <!-- Connection indicator -->
      <div class="flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full" :class="stateColors[connectionState]" />
        <span>{{ stateLabels[connectionState] }}</span>
      </div>

      <!-- FPS -->
      <span v-if="connectionState === 'connected'">
        {{ fps }} FPS
      </span>
    </div>

    <div class="flex items-center gap-4">
      <!-- Sessions -->
      <span>Sessions: {{ activeSessions }}/{{ maxSessions }}</span>

      <!-- Controls hint -->
      <span class="text-hollow-muted/60">
        WASD / Arrows — move &nbsp; Space — jump &nbsp; K — attack &nbsp; J — heal
      </span>
    </div>
  </footer>
</template>
