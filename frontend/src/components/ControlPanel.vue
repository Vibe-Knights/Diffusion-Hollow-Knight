<script setup lang="ts">
defineProps<{
  connected: boolean
  upscalerOn: boolean
  interpolationOn: boolean
  interpolationExp: number
  opticalFlowOn: boolean
  nvofAvailable: boolean
  upscalerAvailable: boolean
  interpolationAvailable: boolean
  interpolationModels: string[]
}>()

const emit = defineEmits<{
  'toggle-upscaler': [val: boolean]
  'toggle-interpolation': [val: boolean]
  'change-exp': [val: number]
  'toggle-optical-flow': [val: boolean]
  'connect': []
  'disconnect': []
}>()
</script>

<template>
  <header class="flex items-center justify-between px-6 py-3 bg-hollow-panel border-b border-hollow-border shrink-0">
    <!-- Title -->
    <div class="flex items-center gap-3">
      <h1 class="text-lg font-bold tracking-wide text-hollow-accent">
        ⚔ Diffusion Hollow Knight
      </h1>
    </div>

    <!-- Controls -->
    <div class="flex items-center gap-6">
      <!-- Upscaler toggle -->
      <label
        class="flex items-center gap-2 cursor-pointer select-none"
        :class="{ 'opacity-40 pointer-events-none': !upscalerAvailable }"
      >
        <span class="text-sm text-hollow-muted">Upscaler</span>
        <button
          class="relative w-10 h-5 rounded-full transition-colors duration-200"
          :class="upscalerOn ? 'bg-hollow-accent' : 'bg-hollow-border'"
          @click="emit('toggle-upscaler', !upscalerOn)"
        >
          <span
            class="absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform duration-200"
            :class="{ 'translate-x-5': upscalerOn }"
          />
        </button>
      </label>

      <!-- Optical Flow toggle -->
      <label
        class="flex items-center gap-2 cursor-pointer select-none"
        :class="{ 'opacity-40 pointer-events-none': !nvofAvailable || !upscalerAvailable || !upscalerOn }"
      >
        <span class="text-sm text-hollow-muted">NVOF</span>
        <button
          class="relative w-10 h-5 rounded-full transition-colors duration-200"
          :class="opticalFlowOn ? 'bg-hollow-accent' : 'bg-hollow-border'"
          @click="emit('toggle-optical-flow', !opticalFlowOn)"
        >
          <span
            class="absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform duration-200"
            :class="{ 'translate-x-5': opticalFlowOn }"
          />
        </button>
      </label>

      <!-- Interpolation toggle -->
      <label
        class="flex items-center gap-2 cursor-pointer select-none"
        :class="{ 'opacity-40 pointer-events-none': !interpolationAvailable }"
      >
        <span class="text-sm text-hollow-muted">Interpolation</span>
        <button
          class="relative w-10 h-5 rounded-full transition-colors duration-200"
          :class="interpolationOn ? 'bg-hollow-accent' : 'bg-hollow-border'"
          @click="emit('toggle-interpolation', !interpolationOn)"
        >
          <span
            class="absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform duration-200"
            :class="{ 'translate-x-5': interpolationOn }"
          />
        </button>
      </label>

      <!-- Interpolation exp -->
      <div
        class="flex items-center gap-2"
        :class="{ 'opacity-40 pointer-events-none': !interpolationAvailable || !interpolationOn }"
      >
        <span class="text-sm text-hollow-muted">Exp</span>
        <div class="flex rounded overflow-hidden border border-hollow-border">
          <button
            v-for="v in [1, 2, 3]"
            :key="v"
            class="px-2 py-0.5 text-xs font-mono transition-colors"
            :class="interpolationExp === v ? 'bg-hollow-accent text-hollow-bg' : 'bg-hollow-panel text-hollow-muted hover:bg-hollow-border'"
            @click="emit('change-exp', v)"
          >
            x{{ Math.pow(2, v) }}
          </button>
        </div>
      </div>

      <!-- Connect / Disconnect -->
      <button
        v-if="!connected"
        class="px-4 py-1.5 text-sm font-medium rounded bg-hollow-accent text-hollow-bg hover:brightness-110 transition"
        @click="emit('connect')"
      >
        Connect
      </button>
      <button
        v-else
        class="px-4 py-1.5 text-sm font-medium rounded bg-red-600/80 text-white hover:bg-red-500 transition"
        @click="emit('disconnect')"
      >
        Disconnect
      </button>
    </div>
  </header>
</template>
