'use client'

import { useState, useEffect, useRef } from 'react'
import { useParams } from 'next/navigation'
import { SocketManager } from '@/lib/socket'
import { MediasoupClient } from '@/lib/mediasoup-client'
import { io, Socket } from 'socket.io-client'

export default function Viewer() {
  const params = useParams()
  const roomId = params.roomId as string

  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState('')
  const [connectionState, setConnectionState] = useState('connecting')
  const [streamAvailable, setStreamAvailable] = useState(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const socketRef = useRef<SocketManager | null>(null)
  const sfuSocketRef = useRef<Socket | null>(null)
  const mediasoupClientRef = useRef<MediasoupClient | null>(null)
  const remoteStreamsRef = useRef<Map<string, MediaStream>>(new Map())
  const consumersRef = useRef<Set<string>>(new Set())

  useEffect(() => {
    const initializeViewer = async () => {
      try {
        setError('')
        setConnectionState('connecting')
        console.log('Initializing viewer for room:', roomId)

        // Initialize main signaling socket (Flask backend)
        socketRef.current = new SocketManager()
        await socketRef.current.connect()
        
        // Join room
        const response = await socketRef.current.joinRoom(roomId)
        
        // Initialize SFU connection (Mediasoup server)
        const sfuUrl = response.mediasoupUrl || 'http://localhost:3001'
        sfuSocketRef.current = io(sfuUrl)
        
        await new Promise((resolve) => {
          sfuSocketRef.current!.on('connect', resolve)
        })

        // Initialize Mediasoup client
        mediasoupClientRef.current = new MediasoupClient(roomId)
        await mediasoupClientRef.current.initialize(sfuSocketRef.current!)
        
        // Join room in SFU server
        await new Promise((resolve, reject) => {
          sfuSocketRef.current!.emit('join-room', { roomId }, (response: any) => {
            if (response.success) {
              resolve(response)
            } else {
              reject(new Error(response.error))
            }
          })
        })

        // Create consumer transport
        await mediasoupClientRef.current.createConsumerTransport()

        console.log('Viewer initialized successfully')
        setIsConnected(true)
        setConnectionState('connected')

        // Set up event handlers
        setupEventHandlers()

      } catch (err) {
        setError(`Failed to join room: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setConnectionState('error')
        console.error('Viewer initialization error:', err)
      }
    }

    const setupEventHandlers = () => {
      if (!socketRef.current || !sfuSocketRef.current || !mediasoupClientRef.current) return

      // Main signaling socket handlers
      socketRef.current.onStreamingStarted(() => {
        console.log('Host started streaming')
        setStreamAvailable(true)
      })

      socketRef.current.onStreamingStopped(() => {
        console.log('Host stopped streaming')
        setStreamAvailable(false)
        // Clean up streams
        remoteStreamsRef.current.clear()
        if (videoRef.current) {
          videoRef.current.srcObject = null
        }
      })

      socketRef.current.onHostDisconnected(() => {
        setError('Host disconnected')
        setStreamAvailable(false)
        setConnectionState('disconnected')
      })

      // SFU socket handlers
      sfuSocketRef.current!.on('new-producer', async (data) => {
        const { producerId, kind } = data
        console.log(`New producer available: ${producerId} (${kind})`)
        
        try {
          // Check if we already have this consumer
          if (consumersRef.current.has(producerId)) {
            console.log(`Already consuming ${producerId}`)
            return
          }

          const stream = await mediasoupClientRef.current!.consume(producerId, kind)
          if (stream) {
            consumersRef.current.add(producerId)
            
            // Add stream to our collection
            remoteStreamsRef.current.set(producerId, stream)
            
            // If this is video, display it
            if (kind === 'video' && videoRef.current) {
              videoRef.current.srcObject = stream
              console.log('Video stream attached to video element')
            }
            
            // If this is audio, create audio element or add to existing stream
            if (kind === 'audio') {
              // If we already have a video stream, add audio track to it
              const existingVideoStream = Array.from(remoteStreamsRef.current.values())
                .find(s => s.getVideoTracks().length > 0)
              
              if (existingVideoStream) {
                stream.getAudioTracks().forEach(track => {
                  existingVideoStream.addTrack(track)
                })
                if (videoRef.current) {
                  videoRef.current.srcObject = existingVideoStream
                }
              } else {
                // Create new MediaStream with audio
                const audioOnlyStream = new MediaStream(stream.getAudioTracks())
                if (videoRef.current) {
                  videoRef.current.srcObject = audioOnlyStream
                }
              }
              console.log('Audio stream processed')
            }
          }
        } catch (err) {
          console.error(`Failed to consume ${kind} producer ${producerId}:`, err)
        }
      })

      sfuSocketRef.current!.on('producer-closed', (data) => {
        const { consumerId } = data
        console.log(`Producer closed: ${consumerId}`)
        
        // Clean up consumer
        consumersRef.current.delete(consumerId)
        remoteStreamsRef.current.delete(consumerId)
        
        // If no more streams, clear video element
        if (remoteStreamsRef.current.size === 0 && videoRef.current) {
          videoRef.current.srcObject = null
        }
      })

      sfuSocketRef.current!.on('host-disconnected', () => {
        setError('Host disconnected from SFU')
        setStreamAvailable(false)
        setConnectionState('disconnected')
      })
    }

    if (roomId) {
      initializeViewer()
    }

    return () => {
      // Cleanup
      if (mediasoupClientRef.current) {
        mediasoupClientRef.current.stopConsuming()
      }
      
      if (socketRef.current) {
        socketRef.current.disconnect()
      }
      
      if (sfuSocketRef.current) {
        sfuSocketRef.current.disconnect()
      }
      
      remoteStreamsRef.current.clear()
      consumersRef.current.clear()
    }
  }, [roomId])

  const getConnectionStatusColor = () => {
    switch (connectionState) {
      case 'connected': return 'bg-green-500'
      case 'connecting': return 'bg-yellow-500'
      case 'error':
      case 'disconnected': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getConnectionStatusText = () => {
    switch (connectionState) {
      case 'connected': return streamAvailable ? 'Watching Live Stream (SFU)' : 'Connected - Waiting for Stream'
      case 'connecting': return 'Connecting to SFU...'
      case 'error': return 'Connection Error'
      case 'disconnected': return 'Disconnected'
      default: return 'Unknown'
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">
            SFU Viewer
          </h1>
          
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Room ID</h3>
              <code className="bg-gray-200 px-2 py-1 rounded text-sm font-mono">
                {roomId}
              </code>
            </div>
            
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Status</h3>
              <div className="text-sm">
                <div className={`inline-block w-2 h-2 rounded-full mr-2 ${getConnectionStatusColor()}`} />
                {getConnectionStatusText()}
              </div>
            </div>
          </div>

          <div className="text-sm text-gray-600 text-center">
            <p><strong>SFU Mode:</strong> Optimized streaming via Mediasoup server</p>
            <p>Low latency, high quality viewing experience</p>
          </div>
        </div>

        <div className="bg-black rounded-lg overflow-hidden relative">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            controls
            className="w-full h-auto object-contain"
            style={{ backgroundColor: '#000', minHeight: '400px' }}
            onLoadedData={() => {
              console.log('Video loaded and ready to play')
            }}
            onError={(e) => {
              console.error('Video error:', e)
            }}
          />
          
          {!streamAvailable && connectionState === 'connected' && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4">⏳</div>
                <div>Waiting for host to start streaming...</div>
                <div className="text-sm mt-2 opacity-75">Connected to SFU server</div>
              </div>
            </div>
          )}
          
          {connectionState === 'connecting' && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4">🔄</div>
                <div>Connecting to SFU server...</div>
                <div className="text-sm mt-2 opacity-75">Please wait</div>
              </div>
            </div>
          )}
          
          {(connectionState === 'error' || connectionState === 'disconnected') && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4">❌</div>
                <div>Connection failed</div>
                <div className="text-sm mt-2 opacity-75">Please refresh the page</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}