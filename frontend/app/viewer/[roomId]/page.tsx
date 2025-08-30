'use client'

import { useState, useEffect, useRef } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { WebRTCPeer } from '@/lib/webrtc'
import { SocketManager } from '@/lib/socket'

export default function Viewer() {
  const params = useParams()
  const router = useRouter()
  const roomId = params.roomId as string

  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')
  const [connectionState, setConnectionState] = useState('')
  const [roomExists, setRoomExists] = useState(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const webrtcRef = useRef<WebRTCPeer | null>(null)
  const socketRef = useRef<SocketManager | null>(null)
  const isConnectedRef = useRef(false)

  useEffect(() => {
    if (!roomId) {
      setError('Invalid room ID')
      setIsLoading(false)
      return
    }

    const initializeViewer = async () => {
      try {
        socketRef.current = new SocketManager()
        await socketRef.current.connect()

        const roomInfo = await socketRef.current.checkRoom(roomId)
        if (!roomInfo.exists) {
          setError('Room not found or stream has ended')
          setIsLoading(false)
          return
        }

        setRoomExists(true)
        await socketRef.current.joinRoom(roomId)

        webrtcRef.current = new WebRTCPeer()

        webrtcRef.current.onRemoteStream((stream) => {
          console.log('Viewer received remote stream:', stream.getTracks().length, 'tracks')
          console.log('Video ref current exists:', !!videoRef.current)
          
          // Only accept the first stream
          if (isConnectedRef.current) {
            console.log('Already connected, ignoring additional stream')
            return
          }
          
          // Mark as connected immediately to prevent multiple streams
          setIsConnected(true)
          isConnectedRef.current = true
          setIsLoading(false)
          console.log('Viewer connected successfully!')
          
          // Set video source with retry logic
          const setVideoSource = () => {
            if (videoRef.current) {
              videoRef.current.srcObject = stream
              console.log('Set video srcObject successfully')
              return true
            }
            return false
          }
          
          // Try immediately, then retry if needed
          if (!setVideoSource()) {
            console.log('Video ref is null, retrying after 100ms...')
            setTimeout(() => {
              if (!setVideoSource()) {
                console.log('Video ref still null after retry')
              }
            }, 100)
          }
        })

        webrtcRef.current.onIceCandidate((candidate) => {
          if (socketRef.current) {
            socketRef.current.sendIceCandidate(candidate)
          }
        })

        socketRef.current.onOffer(async (data) => {
          console.log('Viewer received offer from host:', data.from)
          if (webrtcRef.current) {
            try {
              const answer = await webrtcRef.current.createAnswer(data.offer)
              console.log('Viewer created answer, sending to host')
              socketRef.current!.sendAnswer(answer)
            } catch (err) {
              console.error('Failed to create answer:', err)
            }
          }
        })

        socketRef.current.onIceCandidate(async (data) => {
          console.log('Viewer received ICE candidate from host:', data.from)
          if (webrtcRef.current) {
            try {
              await webrtcRef.current.addIceCandidate(data.candidate)
              console.log('Viewer added ICE candidate successfully')
            } catch (err) {
              console.error('Failed to add ICE candidate:', err)
            }
          }
        })

        socketRef.current.onHostDisconnected(() => {
          setError('Host has disconnected')
          setIsConnected(false)
        })

        const updateConnectionState = () => {
          if (webrtcRef.current) {
            const state = webrtcRef.current.getConnectionState()
            setConnectionState(state)
            
            if (state === 'failed' || state === 'disconnected') {
              setError('Connection failed. Please try refreshing.')
              setIsConnected(false)
            }
          }
        }
        
        const interval = setInterval(updateConnectionState, 1000)
        
        setTimeout(() => {
          if (!isConnectedRef.current) {
            console.log('Connection timeout - no stream received in 10 seconds')
            setError('Failed to connect to stream. The host may not be streaming yet.')
            setIsLoading(false)
          } else {
            console.log('Connection timeout bypassed - already connected')
          }
        }, 10000)

        return () => clearInterval(interval)

      } catch (err) {
        console.error('Viewer initialization error:', err)
        setError('Failed to connect to room')
        setIsLoading(false)
      }
    }

    initializeViewer()

    return () => {
      if (webrtcRef.current) {
        webrtcRef.current.close()
      }
      if (socketRef.current) {
        socketRef.current.disconnect()
      }
    }
  }, [roomId, isConnected])

  const goBack = () => {
    router.push('/')
  }

  const refreshConnection = () => {
    window.location.reload()
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
          <div>Connecting to stream...</div>
          <div className="text-sm text-gray-400 mt-2">Room: {roomId}</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="bg-white rounded-lg p-8 max-w-md w-full text-center">
          <div className="text-red-600 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Connection Error</h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <div className="space-y-3">
            <button
              onClick={refreshConnection}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition-colors"
            >
              Try Again
            </button>
            <button
              onClick={goBack}
              className="w-full bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-4 rounded-lg transition-colors"
            >
              Go Back
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-3xl font-bold text-gray-800">
              Viewing Stream
            </h1>
            <button
              onClick={goBack}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              Leave
            </button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Room ID</h3>
              <code className="bg-gray-200 px-2 py-1 rounded text-sm font-mono">
                {roomId}
              </code>
            </div>
            
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Status</h3>
              <div className="text-sm">
                <div className={`inline-block w-2 h-2 rounded-full mr-2 ${
                  isConnected ? 'bg-green-500' : 'bg-yellow-500'
                }`} />
                {isConnected ? 'Connected' : 'Connecting...'}
              </div>
              {connectionState && (
                <div className="text-xs text-gray-600 mt-1">
                  Connection: {connectionState}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="bg-black rounded-lg overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-auto max-h-96 object-contain"
            style={{ backgroundColor: '#000' }}
          />
          {!isConnected && !error && (
            <div className="aspect-video flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="animate-pulse text-6xl mb-4">📺</div>
                <div>Waiting for stream to start...</div>
                <div className="text-sm text-gray-500 mt-2">
                  Make sure the host has started streaming
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}