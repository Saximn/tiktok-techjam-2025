'use client'

import { useState, useEffect, useRef } from 'react'
import { SocketManager } from '@/lib/socket'
import { MediasoupClient } from '@/lib/mediasoup-client'
import { io, Socket } from 'socket.io-client'

export default function Host() {
  const [roomId, setRoomId] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [viewerCount, setViewerCount] = useState(0)
  const [error, setError] = useState('')
  const [connectionState, setConnectionState] = useState('')

  const videoRef = useRef<HTMLVideoElement>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const socketRef = useRef<SocketManager | null>(null)
  const sfuSocketRef = useRef<Socket | null>(null)
  const mediasoupClientRef = useRef<MediasoupClient | null>(null)
  const isStreamingRef = useRef(false)

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
    }
  }, [])

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

    } catch (err) {
      setError('Failed to start streaming. Please check camera/microphone permissions.')
      setConnectionState('error')
      console.error('Streaming error:', err)
    }
  }

  const stopStreaming = async () => {
    try {
      console.log('Stopping SFU streaming...')
      
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

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
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
          </div>

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