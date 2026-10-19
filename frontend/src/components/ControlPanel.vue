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
  upscaleModels: string[]
  flowBackends: string[]
  flowBackend: string
  upscaleModel: string
  interpolationModel: string
  engineBackend?: string
  availableSeeds?: Array<{ id: string; name: string }>
  activeSeed?: string
  fps?: number
  stepTimeMs?: number
  samplingSteps?: number
}>()

const emit = defineEmits<{
  'toggle-upscaler': [val: boolean]
  'toggle-interpolation': [val: boolean]
  'change-exp': [val: number]
  'toggle-optical-flow': [val: boolean]
  'change-flow-backend': [val: string]
  'change-upscale-model': [val: string]
  'change-interpolation-model': [val: string]
  'change-seed': [val: string]
  'change-steps': [val: number]
  'purge-buffer': []
  'clear-cache': []
  'connect': []
  'disconnect': []
}>()
</script>

<template>
  <div>
    <header class="bg-surface-container-lowest/90 backdrop-blur-xl border-b border-outline-variant/20 shadow-[0_0_15px_rgba(0,0,0,0.5)] flex justify-between items-center h-16 px-gutter w-full z-50 fixed top-0 left-0">
      <div class="flex items-center gap-4">
        <img alt="Vibe Knights Authentic Logo" class="w-8 h-8 object-contain rounded-full" src="/assets/logo.png">
        <span class="font-headline-md text-headline-md text-on-surface tracking-tighter glow-text">VIBE KNIGHTS</span>
        <div class="h-4 w-[1px] bg-outline-variant mx-2"></div>
        <div class="flex items-center gap-2">
          <div class="soul-orb" :class="{ 'opacity-40': !connected }"></div>
          <span class="font-label-caps text-label-caps text-on-surface">
            Engine: {{ connected ? (engineBackend === 'webgpu' ? 'WEBGPU (GPU)' : 'WASM (CPU FALLBACK)') : 'OFFLINE' }}
          </span>
        </div>
        <div v-if="connected" class="flex items-center gap-2 px-3 py-1 rounded-full bg-surface-container-high border border-outline-variant/30 text-xs font-mono shadow-sm">
          <span class="inline-block w-2 h-2 rounded-full" :class="(fps ?? 0) >= 15 ? 'bg-emerald-400' : (fps ?? 0) >= 8 ? 'bg-amber-400' : 'bg-rose-400 animate-pulse'"></span>
          <span class="font-bold text-on-surface">{{ (fps ?? 0) > 0 ? (fps ?? 0).toFixed(1) : '--' }} FPS</span>
          <span class="text-on-surface-variant text-[11px]">({{ (stepTimeMs ?? 0) > 0 ? stepTimeMs : '--' }} ms)</span>
        </div>
        <div v-if="connected && engineBackend === 'wasm'" class="hidden lg:flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 text-[11px] font-mono">
          <span class="material-symbols-outlined text-[14px] text-amber-400">info</span>
          <span>Open in Chrome/Edge for 20 FPS WebGPU</span>
        </div>
      </div>
      <div class="flex items-center gap-6">
        <button 
          @click="connected ? emit('disconnect') : emit('connect')"
          class="bg-surface-bright text-on-surface border border-outline-variant/30 font-label-caps text-label-caps px-6 py-2 rounded hover:bg-surface-variant transition-colors duration-300 active:scale-95 transition-transform flex items-center gap-2"
        >
          <span class="material-symbols-outlined text-[16px]">{{ connected ? 'stop_circle' : 'play_circle' }}</span>
          {{ connected ? 'STOP GAME' : 'START GAME' }}
        </button>
        <div class="flex gap-4">
          <button class="text-on-surface-variant hover:text-on-surface transition-colors duration-300 active:scale-95 transition-transform" title="WebGPU / WASM Mode">
            <span class="material-symbols-outlined">memory</span>
          </button>
          <button class="text-on-surface-variant hover:text-on-surface transition-colors duration-300 active:scale-95 transition-transform">
            <span class="material-symbols-outlined">settings</span>
          </button>
        </div>
      </div>
    </header>

    <nav class="bg-surface-container-lowest border-r border-outline-variant/10 fixed left-0 top-16 h-[calc(100vh-64px)] w-64 flex flex-col py-md z-40 hidden md:flex">
      <div class="flex-1 overflow-y-auto px-gutter flex flex-col gap-2">
        <div class="border-outline-variant/10">
          <div class="flex flex-col gap-4 px-2">
            
            <!-- Starting Seed Scene -->
            <div class="flex flex-col gap-2">
              <span class="font-body-md text-body-md text-on-surface">Starting Scene</span>
              <select 
                :value="activeSeed || 'seed_1'"
                @change="emit('change-seed', ($event.target as HTMLSelectElement).value)"
                class="bg-surface-container-high rounded p-1.5 border border-outline-variant/20 font-label-caps text-[10px] text-on-surface-variant focus:border-on-surface/50 focus:outline-none"
              >
                <option v-if="!availableSeeds || availableSeeds.length === 0" value="seed_1">Forgotten Crossroads</option>
                <option v-for="s in availableSeeds" :key="s.id" :value="s.id">{{ s.name }}</option>
              </select>
            </div>

            <!-- Neural Upscaler Toggle -->
            <div class="flex flex-col gap-2">
              <span class="font-body-md text-body-md text-on-surface">Upscaler (FastUpscaler)</span>
              <div class="flex bg-surface-container-high rounded p-1 border border-outline-variant/20" :class="{ 'opacity-50 pointer-events-none': !upscalerAvailable }">
                <button @click="emit('toggle-upscaler', false)" class="flex-1 font-label-caps text-[10px] py-1 text-center rounded transition-colors" :class="!upscalerOn ? 'bg-on-surface/10 text-on-surface border border-on-surface/20' : 'text-on-surface-variant hover:text-on-surface'">OFF (72p)</button>
                <button @click="emit('toggle-upscaler', true)" class="flex-1 font-label-caps text-[10px] py-1 text-center rounded transition-colors" :class="upscalerOn ? 'bg-on-surface/10 text-on-surface border border-on-surface/20' : 'text-on-surface-variant hover:text-on-surface'">ON (288p)</button>
              </div>
            </div>
            
            <div class="flex flex-col gap-2">
              <span class="font-body-md text-body-md text-on-surface">Model Weights</span>
              <select 
                :value="upscaleModel || 'FastUpscaler 1.6M'"
                @change="emit('change-upscale-model', ($event.target as HTMLSelectElement).value)"
                class="bg-surface-container-high rounded p-1.5 border border-outline-variant/20 font-label-caps text-[10px] text-on-surface-variant focus:border-on-surface/50 focus:outline-none"
                :disabled="!upscalerAvailable || !upscalerOn"
                :class="{ 'opacity-50': !upscalerAvailable || !upscalerOn }"
              >
                <option value="FastUpscaler 1.6M">FastUpscaler 1.6M (Fused)</option>
                <option v-for="model in upscaleModels" :key="model" :value="model">{{ model }}</option>
              </select>
            </div>
            
            <div class="flex flex-col gap-2">
              <span class="font-body-md text-body-md text-on-surface">Optical flow</span>
              <div class="flex bg-surface-container-high rounded p-1 border border-outline-variant/20">
                <button v-for="backend in (flowBackends.length ? flowBackends : ['Zero', 'NVOF', 'OpenCV', 'IFNet'])" :key="backend"
                  @click="emit('change-flow-backend', backend); emit('toggle-optical-flow', backend !== 'Zero')"
                  class="flex-1 font-label-caps text-[10px] py-1 text-center rounded transition-colors"
                  :class="flowBackend === backend ? 'bg-on-surface/10 text-on-surface border border-on-surface/20' : 'text-on-surface-variant hover:text-on-surface'"
                >{{ backend }}</button>
              </div>
            </div>
            
            <div class="flex flex-col gap-2">
              <span class="font-body-md text-body-md text-on-surface">Interpolation</span>
              <div class="flex bg-surface-container-high rounded p-1 border border-outline-variant/20" :class="{ 'opacity-50 pointer-events-none': !interpolationAvailable }">
                <button @click="emit('toggle-interpolation', false)" class="flex-1 font-label-caps text-[10px] py-1 text-center rounded transition-colors" :class="!interpolationOn ? 'bg-on-surface/10 text-on-surface border border-on-surface/20' : 'text-on-surface-variant hover:text-on-surface'">OFF</button>
                <button v-for="exp in [1, 2, 3]" :key="exp" @click="emit('toggle-interpolation', true); emit('change-exp', exp)" class="flex-1 font-label-caps text-[10px] py-1 text-center rounded transition-colors" :class="interpolationOn && interpolationExp === exp ? 'bg-on-surface/10 text-on-surface border border-on-surface/20' : 'text-on-surface-variant hover:text-on-surface'">x{{ Math.pow(2, exp) }}</button>
              </div>
            </div>
            
            <!-- Sampling Steps (Performance vs Quality) -->
            <div class="flex flex-col gap-2">
              <div class="flex items-center justify-between">
                <span class="font-body-md text-body-md text-on-surface">Sampling Steps</span>
                <span class="text-[10px] text-on-surface-variant font-mono">{{ (samplingSteps ?? 2) === 1 ? 'Ultra Fast (20 FPS)' : (samplingSteps ?? 2) === 2 ? 'High Speed (10 FPS)' : 'High Quality' }}</span>
              </div>
              <div class="flex bg-surface-container-high rounded p-1 border border-outline-variant/20">
                <button
                  @click="emit('change-steps', 1)"
                  class="flex-1 font-label-caps text-[9px] py-1 text-center rounded transition-colors"
                  :class="(samplingSteps ?? 2) === 1 ? 'bg-on-surface/10 text-on-surface border border-on-surface/20' : 'text-on-surface-variant hover:text-on-surface'"
                >1 Step (20 FPS)</button>
                <button
                  @click="emit('change-steps', 2)"
                  class="flex-1 font-label-caps text-[9px] py-1 text-center rounded transition-colors"
                  :class="(samplingSteps ?? 2) === 2 ? 'bg-on-surface/10 text-on-surface border border-on-surface/20' : 'text-on-surface-variant hover:text-on-surface'"
                >2 Steps (10 FPS)</button>
                <button
                  @click="emit('change-steps', 3)"
                  class="flex-1 font-label-caps text-[9px] py-1 text-center rounded transition-colors"
                  :class="(samplingSteps ?? 2) === 3 ? 'bg-on-surface/10 text-on-surface border border-on-surface/20' : 'text-on-surface-variant hover:text-on-surface'"
                >3 Steps (Quality)</button>
              </div>
            </div>

            <!-- Performance & Architecture Metrics -->
            <div class="flex flex-col gap-2">
              <span class="font-body-md text-body-md text-on-surface">Performance & Architecture</span>
              <div class="bg-surface-container-high rounded p-2.5 border border-outline-variant/20 text-[11px] text-on-surface-variant font-mono space-y-1.5">
                <div class="flex justify-between items-center">
                  <span>Frame Rate:</span>
                  <span class="text-on-surface font-bold">{{ (fps ?? 0) > 0 ? (fps ?? 0).toFixed(1) : '--' }} FPS</span>
                </div>
                <div class="flex justify-between items-center">
                  <span>Frame Time:</span>
                  <span class="text-on-surface font-bold">{{ (stepTimeMs ?? 0) > 0 ? stepTimeMs : '--' }} ms</span>
                </div>
                <div class="flex justify-between items-center">
                  <span>Execution:</span>
                  <span class="font-bold" :class="engineBackend === 'webgpu' ? 'text-emerald-400' : 'text-amber-400'">
                    {{ engineBackend === 'webgpu' ? 'WebGPU (Hardware 20 FPS)' : 'WASM (CPU Fallback ~3 FPS)' }}
                  </span>
                </div>
                <div class="flex justify-between items-center">
                  <span>World Model:</span>
                  <span>DIAMOND U-Net (15.1M)</span>
                </div>
                <div class="flex justify-between items-center">
                  <span>Upscaler:</span>
                  <span>{{ upscalerOn ? 'FastUpscaler (288p)' : 'Native (72p Pixelated)' }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="px-gutter mt-auto pb-4 pt-4 border-t border-outline-variant/10">
        <button 
          @click="emit('purge-buffer')"
          class="w-full mb-2 border border-outline-variant/30 text-error font-label-caps text-label-caps py-2 rounded hover:bg-error/10 hover:border-error/50 transition-colors active:scale-95"
        >
          PURGE BUFFER
        </button>
        <button 
          @click="emit('clear-cache')"
          class="w-full mb-3 border border-outline-variant/20 text-on-surface-variant text-[10px] font-label-caps py-1.5 rounded hover:bg-surface-variant/40 hover:text-on-surface transition-colors active:scale-95 flex items-center justify-center gap-1.5"
          title="Clears browser Model Cache and reloads the page"
        >
          <span class="material-symbols-outlined text-[13px]">cleaning_services</span>
          CLEAR CACHE & RELOAD
        </button>
        <div class="flex justify-between">
          <a class="flex items-center gap-2 text-on-surface-variant opacity-70 hover:text-on-surface transition-colors" href="https://github.com/Vibe-Knights/Diffusion-Hollow-Knight" target="_blank">
            <span class="material-symbols-outlined text-[18px]">help_outline</span>
            <span class="font-body-md text-body-md">GitHub</span>
          </a>
          <button class="flex items-center gap-2 text-on-surface-variant opacity-70 hover:text-on-surface transition-colors" @click="emit('purge-buffer')">
            <span class="material-symbols-outlined text-[18px]">restart_alt</span>
            <span class="font-body-md text-body-md">Reset</span>
          </button>
        </div>
      </div>
    </nav>
  </div>
</template>
