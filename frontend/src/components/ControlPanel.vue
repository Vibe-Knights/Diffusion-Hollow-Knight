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
  'purge-buffer': []
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
          <span class="font-label-caps text-label-caps text-on-surface">Engine: {{ connected ? (engineBackend?.toUpperCase() || 'WEBGPU') : 'OFFLINE' }}</span>
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
            
            <div class="flex flex-col gap-2">
              <span class="font-body-md text-body-md text-on-surface">Architecture</span>
              <div class="bg-surface-container-high rounded p-2 border border-outline-variant/20 text-[11px] text-on-surface-variant font-mono space-y-1">
                <div>Model: DIAMOND (15.1M)</div>
                <div>Sampler: Euler (EDM Karras)</div>
                <div>Steps: 3 steps/frame</div>
                <div>Client: onnxruntime-web</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="px-gutter mt-auto pb-4 pt-4 border-t border-outline-variant/10">
        <button 
          @click="emit('purge-buffer')"
          class="w-full mb-4 border border-outline-variant/30 text-error font-label-caps text-label-caps py-2 rounded hover:bg-error/10 hover:border-error/50 transition-colors active:scale-95"
        >
          PURGE BUFFER
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
