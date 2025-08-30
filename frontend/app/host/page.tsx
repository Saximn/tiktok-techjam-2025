'use client'

import { useState, useEffect, useRef } from 'react'
import { WebRTCPeer } from '@/lib/webrtc'
import { SocketManager } from '@/lib/socket'

export default function Host() {
  const [roomId, setRoomId] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [viewerCount, setViewerCount] = useState(0)
  const [error, setError] = useState('')
  const [connectionState, setConnectionState] = useState('')

  const videoRef = useRef<HTMLVideoElement>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const socketRef = useRef<SocketManager | null>(null)
  const viewersRef = useRef<Map<string, WebRTCPeer>>(new Map())
  const isStreamingRef = useRef(false)

  useEffect(() => {
    const initializeSocket = async () => {
      try {
        socketRef.current = new SocketManager()
        await socketRef.current.connect()
        
        const roomId = await socketRef.current.createRoom()
        setRoomId(roomId)

        socketRef.current.onViewerJoined(async (data) => {
          const viewerId = data.userId
          setViewerCount(prev => prev + 1)
          
          console.log('Viewer joined:', viewerId, 'Streaming:', isStreamingRef.current, 'Has stream:', !!localStreamRef.current)
          
          // Check if we already have a connection for this viewer
          if (viewersRef.current.has(viewerId)) {
            console.log('Connection already exists for viewer:', viewerId)
            return
          }
          
          // Create a new WebRTC connection for this viewer if streaming
          if (localStreamRef.current && isStreamingRef.current) {
            try {
              console.log('Creating WebRTC connection for viewer:', viewerId)
              const peerConnection = new WebRTCPeer()
              
              // Clone the local stream for this viewer to ensure unique stream IDs
              const clonedStream = localStreamRef.current.clone()
              clonedStream.getTracks().forEach(track => {
                peerConnection.peerConnection.addTrack(track, clonedStream)
              })
              
              // Setup ICE candidate handling for this specific viewer
              peerConnection.onIceCandidate((candidate) => {
                console.log('Sending ICE candidate to viewer:', viewerId)
                socketRef.current?.getSocket().emit('ice_candidate', {
                  candidate: candidate,
                  to: viewerId  // Target specific viewer
                })
              })
              
              // Create and send offer to specific viewer
              const offer = await peerConnection.createOffer()
              console.log('Sending offer to viewer:', viewerId)
              socketRef.current!.getSocket().emit('offer', {
                offer: offer,
                to: viewerId  // Target specific viewer
              })
              
              // Store the peer connection
              viewersRef.current.set(viewerId, peerConnection)
              
            } catch (err) {
              console.error('Failed to create offer for new viewer:', err)
            }
          } else {
            console.log('Cannot create WebRTC connection - not streaming or no local stream')
          }
        })

        socketRef.current.onViewerLeft((data) => {
          const viewerId = data.userId
          const peerConnection = viewersRef.current.get(viewerId)
          if (peerConnection) {
            peerConnection.close()
            viewersRef.current.delete(viewerId)
          }
          setViewerCount(prev => Math.max(0, prev - 1))
        })

        socketRef.current.onAnswer(async (data) => {
          // Find the peer connection for the viewer who sent this answer
          const viewerId = data.from
          const peerConnection = Array.from(viewersRef.current.entries())
            .find(([id, _]) => id === viewerId)?.[1]
          
          if (peerConnection) {
            console.log('Processing answer from viewer:', viewerId)
            await peerConnection.handleAnswer(data.answer)
          } else {
            console.error('No peer connection found for viewer:', viewerId)
          }
        })

        socketRef.current.onIceCandidate(async (data) => {
          // Add ICE candidate to the specific viewer's connection
          const viewerId = data.from
          const peerConnection = Array.from(viewersRef.current.entries())
            .find(([id, _]) => id === viewerId)?.[1]
          
          if (peerConnection) {
            console.log('Processing ICE candidate from viewer:', viewerId)
            await peerConnection.addIceCandidate(data.candidate)
          } else {
            console.error('No peer connection found for viewer ICE candidate:', viewerId)
          }
        })

      } catch (err) {
        setError('Failed to initialize connection')
        console.error('Socket initialization error:', err)
      }
    }

    initializeSocket()

    return () => {
      // Close all peer connections
      for (const peerConnection of viewersRef.current.values()) {
        peerConnection.close()
      }
      viewersRef.current.clear()
      
      // Stop local stream
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach(track => track.stop())
      }
      
      if (socketRef.current) {
        socketRef.current.disconnect()
      }
    }
  }, [])

  const startStreaming = async () => {
    try {
      setError('')
      console.log('Starting stream...')
      
      // Get local media stream
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      localStreamRef.current = stream
      console.log('Got local stream:', stream.getTracks().length, 'tracks')
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }

      setIsStreaming(true)
      isStreamingRef.current = true
      setConnectionState('streaming')
      console.log('Streaming started successfully')

    } catch (err) {
      setError('Failed to start streaming. Please check camera/microphone permissions.')
      console.error('Streaming error:', err)
    }
  }


  const stopStreaming = () => {
    // Close all peer connections
    for (const peerConnection of viewersRef.current.values()) {
      peerConnection.close()
    }
    viewersRef.current.clear()
    
    // Stop local stream
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop())
      localStreamRef.current = null
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    
    setIsStreaming(false)
    isStreamingRef.current = false
    setConnectionState('')
    setViewerCount(0)
  }

  const copyRoomId = () => {
    navigator.clipboard.writeText(roomId)
  }

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">
            Host Stream
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
                {isStreaming ? 'Live' : 'Offline'}
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
                className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition-colors"
              >
                Start Streaming
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
                <div>Click "Start Streaming" to begin</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}