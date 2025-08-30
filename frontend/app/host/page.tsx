'use client'

import { useState, useEffect, useRef } from 'react'
import { SocketManager, PIIDetectionData, AudioProcessingConfig, PIIEntity } from '@/lib/socket'
import { MediasoupClient } from '@/lib/mediasoup-client'
import { io, Socket } from 'socket.io-client'

export default function Host() {
  const [roomId, setRoomId] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [viewerCount, setViewerCount] = useState(0)
  const [error, setError] = useState('')
  const [connectionState, setConnectionState] = useState('')

  // Audio Processing State
  const [isPIIDetectionEnabled, setIsPIIDetectionEnabled] = useState(false)
  const [piiAlerts, setPiiAlerts] = useState<PIIDetectionData[]>([])
  const [audioProcessingConfig, setAudioProcessingConfig] = useState<AudioProcessingConfig>({
    whisper_model_size: 'base',
    min_confidence: 0.7,
    redaction_type: 'beep',
    enable_audio_redaction: true,
    enable_mouth_blur: true
  })
  const [recentPII, setRecentPII] = useState<PIIEntity[]>([])
  const [totalPIIDetected, setTotalPIIDetected] = useState(0)

  const videoRef = useRef<HTMLVideoElement>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const socketRef = useRef<SocketManager | null>(null)
  const sfuSocketRef = useRef<Socket | null>(null)
  const mediasoupClientRef = useRef<MediasoupClient | null>(null)
  const isStreamingRef = useRef(false)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null)

  useEffect(() => {
    const initializeConnections = async () => {
      try {
        setError('')
        
        // Initialize main signaling socket (Flask backend)
        socketRef.current = new SocketManager()
        await socketRef.current.connect()
        
        // Create room and get room info
        const response = await socketRef.current.createRoom()
        setRoomId(response.roomId)
        
        // Initialize SFU connection (Mediasoup server)
        const sfuUrl = response.mediasoupUrl || 'http://localhost:3001'
        sfuSocketRef.current = io(sfuUrl)
        
        await new Promise((resolve) => {
          sfuSocketRef.current!.on('connect', resolve)
        })

        // Initialize Mediasoup client
        mediasoupClientRef.current = new MediasoupClient(response.roomId)
        await mediasoupClientRef.current.initialize(sfuSocketRef.current!)
        
        // Create room in SFU server
        await new Promise((resolve, reject) => {
          sfuSocketRef.current!.emit('create-room', { roomId: response.roomId }, (response: any) => {
            if (response.success) {
              resolve(response)
            } else {
              reject(new Error(response.error))
            }
          })
        })

        console.log('Host connections initialized successfully')
        setConnectionState('ready')

        // Set up event handlers
        setupEventHandlers()

      } catch (err) {
        setError('Failed to initialize connections')
        console.error('Connection initialization error:', err)
      }
    }

    const setupEventHandlers = () => {
      if (!socketRef.current || !sfuSocketRef.current) return

      // Main signaling socket handlers
      socketRef.current.onViewerJoined((data) => {
        console.log('Viewer joined:', data.userId)
        setViewerCount(data.viewerCount || (prev => prev + 1))
      })

      socketRef.current.onViewerLeft((data) => {
        console.log('Viewer left:', data.userId)
        setViewerCount(data.viewerCount || (prev => Math.max(0, prev - 1)))
      })

      // Audio Processing Event Handlers
      socketRef.current.onPIIDetected((data) => {
        console.log('PII detected:', data.entities)
        
        // Add to alerts (keep last 10)
        setPiiAlerts(prev => [data, ...prev].slice(0, 10))
        
        // Update recent PII entities
        setRecentPII(prev => [...data.entities, ...prev].slice(0, 20))
        
        // Update total count
        setTotalPIIDetected(prev => prev + data.entities.length)
        
        // Apply mouth blur if enabled
        if (data.enable_mouth_blur && data.redaction_intervals) {
          data.redaction_intervals.forEach(([startTime, endTime]) => {
            scheduleBlurMouth(startTime, endTime)
          })
        }
      })

      socketRef.current.onAudioProcessingStarted((data) => {
        console.log('Audio processing started for room:', data.room_id)
        setIsPIIDetectionEnabled(true)
      })

      socketRef.current.onAudioProcessingStopped((data) => {
        console.log('Audio processing stopped for room:', data.room_id)
        setIsPIIDetectionEnabled(false)
      })

      socketRef.current.onAudioProcessingError((error) => {
        console.error('Audio processing error:', error.error)
        setError(`Audio processing error: ${error.error}`)
        setIsPIIDetectionEnabled(false)
      })

      // SFU socket handlers
      sfuSocketRef.current!.on('viewer-joined', (data) => {
        console.log('SFU: Viewer joined', data.viewerId)
      })

      sfuSocketRef.current!.on('viewer-left', (data) => {
        console.log('SFU: Viewer left', data.viewerId)
      })
    }

    initializeConnections()

    return () => {
      // Cleanup
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach(track => track.stop())
      }
      
      if (mediasoupClientRef.current) {
        mediasoupClientRef.current.stopProducing()
      }
      
      if (socketRef.current) {
        socketRef.current.disconnect()
      }
      
      if (sfuSocketRef.current) {
        sfuSocketRef.current.disconnect()
      }
      
      // Audio processing cleanup
      if (audioProcessorRef.current) {
        audioProcessorRef.current.disconnect()
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
    }
  }, [])

  // Audio Processing Functions
  const enablePIIDetection = async () => {
    if (!socketRef.current || !roomId) return
    
    try {
      setError('')
      await socketRef.current.startAudioProcessing(roomId, audioProcessingConfig)
      
      // Set up audio processing from stream
      if (localStreamRef.current) {
        await setupAudioProcessing(localStreamRef.current)
      }
      
      console.log('PII detection enabled')
    } catch (error) {
      console.error('Failed to enable PII detection:', error)
      setError('Failed to enable PII detection')
    }
  }

  const disablePIIDetection = async () => {
    if (!socketRef.current || !roomId) return
    
    try {
      await socketRef.current.stopAudioProcessing(roomId)
      
      // Clean up audio processing
      if (audioProcessorRef.current) {
        audioProcessorRef.current.disconnect()
        audioProcessorRef.current = null
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
      
      console.log('PII detection disabled')
    } catch (error) {
      console.error('Failed to disable PII detection:', error)
    }
  }

  const setupAudioProcessing = async (stream: MediaStream) => {
    try {
      const audioTrack = stream.getAudioTracks()[0]
      if (!audioTrack) return

      // Create audio context
      audioContextRef.current = new AudioContext({ sampleRate: 16000 })
      const source = audioContextRef.current.createMediaStreamSource(stream)
      
      // Create processor for audio chunks
      audioProcessorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1)
      
      audioProcessorRef.current.onaudioprocess = (event) => {
        if (!socketRef.current || !isPIIDetectionEnabled) return
        
        const inputBuffer = event.inputBuffer
        const inputData = inputBuffer.getChannelData(0)
        
        // Convert to 16-bit PCM and base64 encode
        const pcmData = new Int16Array(inputData.length)
        for (let i = 0; i < inputData.length; i++) {
          pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768))
        }
        
        const base64Audio = btoa(String.fromCharCode(...new Uint8Array(pcmData.buffer)))
        
        // Send audio chunk to backend
        socketRef.current.sendAudioChunk(roomId, base64Audio, Date.now())
      }
      
      source.connect(audioProcessorRef.current)
      audioProcessorRef.current.connect(audioContextRef.current.destination)
      
    } catch (error) {
      console.error('Error setting up audio processing:', error)
    }
  }

  const scheduleBlurMouth = (startTime: number, endTime: number) => {
    // This function coordinates with your existing mouth blur system
    console.log(`Scheduling mouth blur from ${startTime}s to ${endTime}s`)
    
    // You can integrate with your existing mouth blur implementation here
    // For example, if you have a mouth blur function:
    // blurMouthDuringInterval(startTime * 1000, endTime * 1000)
    
    // For demonstration, we'll just log it
    setTimeout(() => {
      console.log(`Starting mouth blur at ${startTime}s`)
    }, startTime * 1000)
    
    setTimeout(() => {
      console.log(`Ending mouth blur at ${endTime}s`)
    }, endTime * 1000)
  }

  const clearPIIAlerts = () => {
    setPiiAlerts([])
    setRecentPII([])
    setTotalPIIDetected(0)
  }

  const startStreaming = async () => {
    try {
      setError('')
      setConnectionState('initializing')
      console.log('Starting SFU streaming...')
      
      if (!mediasoupClientRef.current || !socketRef.current) {
        throw new Error('Connections not initialized')
      }

      // Get local media stream
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        }, 
        audio: {
          sampleRate: 48000,
          channelCount: 2
        }
      })
      
      localStreamRef.current = stream
      console.log('Got local stream:', stream.getTracks().length, 'tracks')
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }

      // Create producer transport and start producing
      await mediasoupClientRef.current.createProducerTransport()
      await mediasoupClientRef.current.produce(stream)
      
      // Notify signaling server that streaming has started
      socketRef.current.getSocket().emit('sfu_streaming_started', { roomId })
      
      setIsStreaming(true)
      isStreamingRef.current = true
      setConnectionState('streaming')
      console.log('SFU streaming started successfully')

      // Auto-enable PII detection when streaming starts
      if (audioProcessingConfig.enable_audio_redaction || audioProcessingConfig.enable_mouth_blur) {
        setTimeout(() => {
          enablePIIDetection()
        }, 1000) // Small delay to ensure stream is fully established
      }

    } catch (err) {
      setError('Failed to start streaming. Please check camera/microphone permissions.')
      setConnectionState('error')
      console.error('Streaming error:', err)
    }
  }

  const stopStreaming = async () => {
    try {
      console.log('Stopping SFU streaming...')
      
      // Disable PII detection first
      if (isPIIDetectionEnabled) {
        await disablePIIDetection()
      }
      
      // Stop producing
      if (mediasoupClientRef.current) {
        await mediasoupClientRef.current.stopProducing()
      }
      
      // Stop local stream
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach(track => track.stop())
        localStreamRef.current = null
      }
      
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
      
      // Notify signaling server that streaming has stopped
      if (socketRef.current) {
        socketRef.current.getSocket().emit('sfu_streaming_stopped', { roomId })
      }
      
      setIsStreaming(false)
      isStreamingRef.current = false
      setConnectionState('ready')
      setViewerCount(0)
      console.log('SFU streaming stopped')
    } catch (err) {
      console.error('Error stopping streaming:', err)
    }
  }

  const copyRoomId = () => {
    navigator.clipboard.writeText(roomId)
  }

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">
            SFU Host Stream
          </h1>
          
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Room ID</h3>
              <div className="flex items-center gap-2">
                <code className="bg-gray-200 px-2 py-1 rounded text-sm font-mono">
                  {roomId || 'Generating...'}
                </code>
                <button
                  onClick={copyRoomId}
                  disabled={!roomId}
                  className="bg-blue-600 hover:bg-blue-700 text-white text-xs px-2 py-1 rounded disabled:opacity-50"
                >
                  Copy
                </button>
              </div>
            </div>
            
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Viewers</h3>
              <div className="text-2xl font-bold text-green-600">{viewerCount}</div>
            </div>
            
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Status</h3>
              <div className="text-sm">
                <div className={`inline-block w-2 h-2 rounded-full mr-2 ${
                  isStreaming ? 'bg-green-500' : 'bg-red-500'
                }`} />
                {isStreaming ? 'Live (SFU)' : 'Offline'}
              </div>
              {connectionState && (
                <div className="text-xs text-gray-600 mt-1">
                  Connection: {connectionState}
                </div>
              )}
            </div>

            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">PII Detection</h3>
              <div className="text-sm">
                <div className={`inline-block w-2 h-2 rounded-full mr-2 ${
                  isPIIDetectionEnabled ? 'bg-green-500' : 'bg-red-500'
                }`} />
                {isPIIDetectionEnabled ? 'Active' : 'Inactive'}
              </div>
              <div className="text-xs text-gray-600 mt-1">
                Detected: {totalPIIDetected}
              </div>
            </div>
          </div>

          {/* PII Detection Controls */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-blue-800">🛡️ Privacy Protection (PII Detection)</h3>
              <div className="flex gap-2">
                <button
                  onClick={enablePIIDetection}
                  disabled={isPIIDetectionEnabled || !isStreaming}
                  className="bg-green-600 hover:bg-green-700 text-white text-sm px-3 py-1 rounded disabled:opacity-50"
                >
                  Enable
                </button>
                <button
                  onClick={disablePIIDetection}
                  disabled={!isPIIDetectionEnabled}
                  className="bg-red-600 hover:bg-red-700 text-white text-sm px-3 py-1 rounded disabled:opacity-50"
                >
                  Disable
                </button>
                <button
                  onClick={clearPIIAlerts}
                  className="bg-gray-600 hover:bg-gray-700 text-white text-sm px-3 py-1 rounded"
                >
                  Clear Alerts
                </button>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <label className="block text-blue-700 font-medium mb-1">Model Size:</label>
                <select 
                  value={audioProcessingConfig.whisper_model_size}
                  onChange={(e) => setAudioProcessingConfig(prev => ({
                    ...prev, 
                    whisper_model_size: e.target.value as any
                  }))}
                  className="w-full border border-blue-300 rounded px-2 py-1"
                  disabled={isPIIDetectionEnabled}
                >
                  <option value="tiny">Tiny (Fastest)</option>
                  <option value="base">Base (Balanced)</option>
                  <option value="small">Small (Better Accuracy)</option>
                </select>
              </div>
              
              <div>
                <label className="block text-blue-700 font-medium mb-1">Confidence:</label>
                <select 
                  value={audioProcessingConfig.min_confidence}
                  onChange={(e) => setAudioProcessingConfig(prev => ({
                    ...prev, 
                    min_confidence: parseFloat(e.target.value)
                  }))}
                  className="w-full border border-blue-300 rounded px-2 py-1"
                >
                  <option value={0.5}>Low (0.5)</option>
                  <option value={0.7}>Medium (0.7)</option>
                  <option value={0.9}>High (0.9)</option>
                </select>
              </div>
              
              <div>
                <label className="block text-blue-700 font-medium mb-1">Redaction:</label>
                <select 
                  value={audioProcessingConfig.redaction_type}
                  onChange={(e) => setAudioProcessingConfig(prev => ({
                    ...prev, 
                    redaction_type: e.target.value as any
                  }))}
                  className="w-full border border-blue-300 rounded px-2 py-1"
                >
                  <option value="beep">Beep</option>
                  <option value="silence">Silence</option>
                  <option value="reverse">Reverse</option>
                </select>
              </div>
            </div>
          </div>

          {/* PII Alerts */}
          {piiAlerts.length > 0 && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <h3 className="font-semibold text-red-800 mb-3">🚨 Recent PII Detections</h3>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {piiAlerts.slice(0, 5).map((alert, index) => (
                  <div key={index} className="bg-red-100 border border-red-300 rounded p-2">
                    <div className="text-sm text-red-800">
                      <strong>{new Date(alert.timestamp).toLocaleTimeString()}</strong>
                      {' - '}Found {alert.entities.length} PII entities
                    </div>
                    <div className="text-xs text-red-600 mt-1">
                      {alert.entities.map(entity => `${entity.label}: ${entity.text}`).join(', ')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-4 justify-center mb-6">
            {!isStreaming ? (
              <button
                onClick={startStreaming}
                disabled={connectionState !== 'ready'}
                className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {connectionState === 'ready' ? 'Start SFU Streaming' : 'Initializing...'}
              </button>
            ) : (
              <button
                onClick={stopStreaming}
                className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-lg transition-colors"
              >
                Stop Streaming
              </button>
            )}
          </div>

          <div className="text-sm text-gray-600 text-center">
            <p><strong>SFU Mode:</strong> Scalable streaming via Mediasoup server</p>
            <p>Supports hundreds of concurrent viewers</p>
            <p><strong>Privacy Protection:</strong> Real-time PII detection with audio redaction & mouth blur</p>
          </div>
        </div>

        <div className="bg-black rounded-lg overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-auto max-h-96 object-contain"
            style={{ backgroundColor: '#000' }}
          />
          {!isStreaming && (
            <div className="aspect-video flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4">📹</div>
                <div>Click "Start SFU Streaming" to begin</div>
                <div className="text-sm mt-2 opacity-75">Powered by Mediasoup SFU</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}