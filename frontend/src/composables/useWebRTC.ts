import { ref, type Ref } from 'vue'
import type { ConnectionState, InputMessage, SettingsMessage } from '@/types'

const API_BASE = '/api'

export function useWebRTC() {
  const connectionState: Ref<ConnectionState> = ref('idle')
  const remoteStream: Ref<MediaStream | null> = ref(null)
  const fps: Ref<number> = ref(0)
  const error: Ref<string | null> = ref(null)

  let pc: RTCPeerConnection | null = null
  let dataChannel: RTCDataChannel | null = null
  let frameCount = 0
  let fpsInterval: ReturnType<typeof setInterval> | null = null

  async function connect(): Promise<void> {
    if (pc) {
      await disconnect()
    }

    connectionState.value = 'connecting'
    error.value = null

    try {
      pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      })

      // Create data channel for input
      dataChannel = pc.createDataChannel('inputs', { ordered: true })
      dataChannel.onopen = () => {
        console.log('DataChannel opened')
      }
      dataChannel.onclose = () => {
        console.log('DataChannel closed')
      }

      // Handle incoming video track
      pc.ontrack = (event: RTCTrackEvent) => {
        if (event.streams && event.streams[0]) {
          remoteStream.value = event.streams[0]
        }
      }

      pc.onconnectionstatechange = () => {
        if (!pc) return
        const state = pc.connectionState
        console.log('Connection state:', state)
        if (state === 'connected') {
          connectionState.value = 'connected'
          startFpsCounter()
        } else if (state === 'failed' || state === 'closed') {
          connectionState.value = 'disconnected'
          stopFpsCounter()
        } else if (state === 'disconnected') {
          connectionState.value = 'disconnected'
          stopFpsCounter()
        }
      }

      // Add a transceiver to receive video
      pc.addTransceiver('video', { direction: 'recvonly' })

      // Create offer
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)

      // Send offer to backend
      const response = await fetch(`${API_BASE}/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        }),
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || `Server error: ${response.status}`)
      }

      const answer = await response.json()
      await pc.setRemoteDescription(new RTCSessionDescription(answer))

    } catch (e: any) {
      console.error('WebRTC connect failed:', e)
      error.value = e.message || 'Connection failed'
      connectionState.value = 'error'
    }
  }

  async function disconnect(): Promise<void> {
    stopFpsCounter()
    if (dataChannel) {
      dataChannel.close()
      dataChannel = null
    }
    if (pc) {
      pc.close()
      pc = null
    }
    remoteStream.value = null
    connectionState.value = 'idle'
  }

  function sendInput(msg: InputMessage): void {
    if (dataChannel && dataChannel.readyState === 'open') {
      dataChannel.send(JSON.stringify(msg))
    }
  }

  function sendSettings(msg: SettingsMessage): void {
    if (dataChannel && dataChannel.readyState === 'open') {
      dataChannel.send(JSON.stringify(msg))
    }
  }

  function startFpsCounter(): void {
    frameCount = 0
    fpsInterval = setInterval(() => {
      fps.value = frameCount
      frameCount = 0
    }, 1000)

    // Count frames via video element events — caller should bind onFrameReceived
  }

  function stopFpsCounter(): void {
    if (fpsInterval) {
      clearInterval(fpsInterval)
      fpsInterval = null
    }
    fps.value = 0
  }

  function countFrame(): void {
    frameCount++
  }

  return {
    connectionState,
    remoteStream,
    fps,
    error,
    connect,
    disconnect,
    sendInput,
    sendSettings,
    countFrame,
  }
}
