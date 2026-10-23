import { useState, useRef, useEffect } from 'react'
import { Microphone, Stop, Waveform, ChatCircleDots, SpeakerHigh, Lightning } from '@phosphor-icons/react'
import './App.css'

const WS_URL = import.meta.env.DEV
  ? 'ws://192.168.0.151:8000/api/v1/voice/stream'
  : 'ws://localhost:8000/api/v1/voice/stream'

type AgentState = 'LISTENING' | 'THINKING' | 'SPEAKING'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [agentState, setAgentState] = useState<AgentState>('LISTENING')
  const [messages, setMessages] = useState<Message[]>([])
  const [vadProbability, setVadProbability] = useState(0)
  const [streamingText, setStreamingText] = useState('')

  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const audioQueueRef = useRef<Int16Array[]>([])
  const isPlayingRef = useRef(false)
  const speechEndTimeRef = useRef<number>(0)

  // Connect WebSocket
  const connect = async () => {
    try {
      const ws = new WebSocket(WS_URL)

      ws.onopen = () => {
        console.log('WebSocket connected')
        // Send init message
        ws.send(JSON.stringify({
          type: 'init',
          voice: 'af_heart'
        }))
      }

      ws.onmessage = async (event) => {
        if (event.data instanceof Blob) {
          // Binary audio data
          const audioReceiveTime = Date.now()
          const ttfa = speechEndTimeRef.current > 0 ? audioReceiveTime - speechEndTimeRef.current : 0
          console.log(`🔊 Audio received, TTFA: ${ttfa}ms from speech end`)
          console.log('Received audio blob, size:', event.data.size)
          const arrayBuffer = await event.data.arrayBuffer()
          const pcm16 = new Int16Array(arrayBuffer)
          audioQueueRef.current.push(pcm16)
          console.log('Audio queue length:', audioQueueRef.current.length, 'isPlaying:', isPlayingRef.current)
          if (!isPlayingRef.current) {
            playAudioQueue()
          }
        } else {
          // JSON message
          const data = JSON.parse(event.data)
          console.log('Received message:', data.type, data)
          handleServerMessage(data)
        }
      }

      ws.onclose = (event) => {
        console.log('WebSocket disconnected, code:', event.code, 'reason:', event.reason)
        setIsConnected(false)
        stopRecording()
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      wsRef.current = ws
      setIsConnected(true)
    } catch (error) {
      console.error('Connection error:', error)
      alert('Failed to connect. Make sure the server is running on localhost:8000')
    }
  }

  const handleServerMessage = (data: any) => {
    switch (data.type) {
      case 'session':
        console.log('Session created:', data.session_id)
        break
      case 'agent_state':
        setAgentState(data.state)
        break
      case 'vad_state':
        if (data.state === 'speech_start') {
          setVadProbability(data.probability)
        } else if (data.state === 'speech_end') {
          speechEndTimeRef.current = Date.now()
          console.log('🎤 Speech ended at:', speechEndTimeRef.current)
        }
        break
      case 'transcript':
        const transcriptTime = Date.now()
        const sttLatency = transcriptTime - speechEndTimeRef.current
        console.log(`📝 Transcript received: ${sttLatency}ms after speech end`)
        setMessages(prev => [...prev, {
          role: 'user',
          content: data.text,
          timestamp: transcriptTime
        }])
        break
      case 'llm_chunk':
        setStreamingText(prev => prev + data.token)
        break
      case 'turn_end':
        if (streamingText) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: streamingText,
            timestamp: Date.now()
          }])
          setStreamingText('')
        }
        break
      case 'interrupted':
        setStreamingText('')
        audioQueueRef.current = []
        // Don't close AudioContext - just stop playback
        isPlayingRef.current = false
        break
    }
  }

  const playAudioQueue = async () => {
    console.log('playAudioQueue called, queue length:', audioQueueRef.current.length)
    if (audioQueueRef.current.length === 0) {
      console.log('Queue empty, stopping playback')
      isPlayingRef.current = false
      return
    }

    isPlayingRef.current = true

    if (!audioContextRef.current) {
      console.log('Creating new AudioContext')
      audioContextRef.current = new AudioContext({ sampleRate: 24000 })
    }

    const ctx = audioContextRef.current
    const pcm16 = audioQueueRef.current.shift()!
    console.log('Playing audio chunk, samples:', pcm16.length)

    // Convert Int16 to Float32
    const float32 = new Float32Array(pcm16.length)
    for (let i = 0; i < pcm16.length; i++) {
      float32[i] = pcm16[i] / 32768.0
    }

    const buffer = ctx.createBuffer(1, float32.length, 24000)
    buffer.copyToChannel(float32, 0)

    const source = ctx.createBufferSource()
    source.buffer = buffer
    source.connect(ctx.destination)

    source.onended = () => {
      console.log('Audio chunk ended, continuing queue')
      playAudioQueue()
    }

    source.start()
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      })

      const audioContext = new AudioContext({ sampleRate: 16000 })
      const source = audioContext.createMediaStreamSource(stream)
      const processor = audioContext.createScriptProcessor(512, 1, 1)

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return

        const inputData = e.inputBuffer.getChannelData(0)
        const pcm16 = new Int16Array(inputData.length)

        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]))
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
        }

        wsRef.current.send(pcm16.buffer)
      }

      source.connect(processor)
      processor.connect(audioContext.destination)

      mediaStreamRef.current = stream
      processorRef.current = processor
      setIsRecording(true)
    } catch (error) {
      console.error('Microphone access error:', error)
      alert('Please allow microphone access to use voice chat')
    }
  }

  const stopRecording = () => {
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }
    setIsRecording(false)
  }

  const disconnect = () => {
    stopRecording()
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
  }

  const interrupt = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'interrupt' }))
    }
  }

  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [])

  const getStateColor = () => {
    switch (agentState) {
      case 'LISTENING': return 'text-green-400'
      case 'THINKING': return 'text-amber-400'
      case 'SPEAKING': return 'text-blue-400'
    }
  }

  const getStateIcon = () => {
    switch (agentState) {
      case 'LISTENING': return <Waveform className="animate-pulse" />
      case 'THINKING': return <Lightning className="animate-spin" />
      case 'SPEAKING': return <SpeakerHigh className="animate-bounce" />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="border-b border-white/10 backdrop-blur-xl bg-white/5">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <ChatCircleDots size={24} weight="bold" />
            </div>
            <div>
              <h1 className="text-xl font-bold">TalkBuddy</h1>
              <p className="text-xs text-slate-400">Voice Testing Interface</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'} animate-pulse`}></div>
              <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>

            {!isConnected ? (
              <button
                onClick={connect}
                className="px-6 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 transition-colors font-medium"
              >
                Connect
              </button>
            ) : (
              <button
                onClick={disconnect}
                className="px-6 py-2 rounded-lg bg-red-600 hover:bg-red-700 transition-colors font-medium"
              >
                Disconnect
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Chat Panel */}
          <div className="lg:col-span-2 space-y-6">

            {/* Messages */}
            <div className="rounded-2xl border border-white/10 backdrop-blur-xl bg-white/5 p-6 h-[500px] flex flex-col">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <ChatCircleDots size={20} />
                Conversation
              </h2>

              <div className="flex-1 overflow-y-auto space-y-4 mb-4">
                {messages.length === 0 && (
                  <div className="h-full flex items-center justify-center text-slate-400">
                    <p>No messages yet. Start talking!</p>
                  </div>
                )}

                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                        msg.role === 'user'
                          ? 'bg-purple-600 text-white'
                          : 'bg-white/10 text-slate-100'
                      }`}
                    >
                      <p className="text-sm">{msg.content}</p>
                    </div>
                  </div>
                ))}

                {streamingText && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-white/10 text-slate-100">
                      <p className="text-sm">{streamingText}<span className="animate-pulse">▊</span></p>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="flex items-center gap-4">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    disabled={!isConnected}
                    className="flex-1 py-4 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium flex items-center justify-center gap-2"
                  >
                    <Microphone size={20} weight="bold" />
                    Start Talking
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="flex-1 py-4 rounded-xl bg-red-600 hover:bg-red-700 transition-colors font-medium flex items-center justify-center gap-2"
                  >
                    <Stop size={20} weight="fill" />
                    Stop Recording
                  </button>
                )}

                <button
                  onClick={interrupt}
                  disabled={!isConnected || agentState !== 'SPEAKING'}
                  className="px-6 py-4 rounded-xl bg-amber-600 hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  Interrupt
                </button>
              </div>
            </div>
          </div>

          {/* Status Panel */}
          <div className="space-y-6">

            {/* Agent State */}
            <div className="rounded-2xl border border-white/10 backdrop-blur-xl bg-white/5 p-6">
              <h2 className="text-lg font-semibold mb-4">Agent Status</h2>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">State</span>
                  <div className={`flex items-center gap-2 ${getStateColor()} font-medium`}>
                    {getStateIcon()}
                    {agentState}
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Recording</span>
                  <span className={isRecording ? 'text-red-400' : 'text-slate-500'}>
                    {isRecording ? '● REC' : 'Inactive'}
                  </span>
                </div>

                {vadProbability > 0 && (
                  <div className="pt-4 border-t border-white/10">
                    <span className="text-slate-400 text-sm">VAD Probability</span>
                    <div className="mt-2 h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-300"
                        style={{ width: `${vadProbability * 100}%` }}
                      />
                    </div>
                    <p className="text-xs text-slate-500 mt-1">{(vadProbability * 100).toFixed(1)}%</p>
                  </div>
                )}
              </div>
            </div>

            {/* Instructions */}
            <div className="rounded-2xl border border-white/10 backdrop-blur-xl bg-white/5 p-6">
              <h2 className="text-lg font-semibold mb-4">How to Use</h2>
              <ol className="space-y-3 text-sm text-slate-300">
                <li className="flex gap-3">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-600 flex items-center justify-center text-xs">1</span>
                  <span>Click <strong>Connect</strong> to establish WebSocket connection</span>
                </li>
                <li className="flex gap-3">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-600 flex items-center justify-center text-xs">2</span>
                  <span>Click <strong>Start Talking</strong> and allow microphone access</span>
                </li>
                <li className="flex gap-3">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-600 flex items-center justify-center text-xs">3</span>
                  <span>Speak naturally - VAD will detect your speech</span>
                </li>
                <li className="flex gap-3">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-600 flex items-center justify-center text-xs">4</span>
                  <span>Click <strong>Interrupt</strong> to barge-in during agent response</span>
                </li>
              </ol>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
