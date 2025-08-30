import { io, Socket } from 'socket.io-client'

export class SocketManager {
  private socket: Socket
  private serverUrl: string

  constructor(serverUrl = 'http://localhost:5000') {
    this.serverUrl = serverUrl
    this.socket = io(serverUrl)
  }

  connect(): Promise<string> {
    return new Promise((resolve) => {
      this.socket.on('connected', (data) => {
        resolve(data.userId)
      })
    })
  }

  createRoom(): Promise<{ roomId: string; mediasoupUrl?: string }> {
    return new Promise((resolve) => {
      this.socket.emit('create_room')
      this.socket.on('room_created', (data) => {
        resolve(data)
      })
    })
  }

  joinRoom(roomId: string): Promise<{ roomId: string; mediasoupUrl?: string }> {
    return new Promise((resolve, reject) => {
      this.socket.emit('join_room', { roomId })
      this.socket.on('joined_room', (data) => {
        resolve(data)
      })
      this.socket.on('error', (error) => {
        reject(error.message)
      })
    })
  }

  checkRoom(roomId: string): Promise<{ exists: boolean; viewerCount?: number }> {
    return new Promise((resolve) => {
      this.socket.emit('get_room_info', { roomId })
      this.socket.on('room_info', (data) => {
        resolve(data)
      })
    })
  }

  sendOffer(offer: RTCSessionDescriptionInit) {
    this.socket.emit('offer', { offer })
  }

  sendAnswer(answer: RTCSessionDescriptionInit) {
    this.socket.emit('answer', { answer })
  }

  sendIceCandidate(candidate: RTCIceCandidate) {
    this.socket.emit('ice_candidate', { candidate })
  }

  onOffer(callback: (data: { offer: RTCSessionDescriptionInit; from: string }) => void) {
    this.socket.on('offer', callback)
  }

  onAnswer(callback: (data: { answer: RTCSessionDescriptionInit; from: string }) => void) {
    this.socket.on('answer', callback)
  }

  onIceCandidate(callback: (data: { candidate: RTCIceCandidateInit; from: string }) => void) {
    this.socket.on('ice_candidate', callback)
  }

  onViewerJoined(callback: (data: { userId: string; viewerCount?: number }) => void) {
    this.socket.on('viewer_joined', callback)
  }

  onViewerLeft(callback: (data: { userId: string; viewerCount?: number }) => void) {
    this.socket.on('viewer_left', callback)
  }

  onHostDisconnected(callback: () => void) {
    this.socket.on('host_disconnected', callback)
  }

  onStreamingStarted(callback: () => void) {
    this.socket.on('streaming_started', callback)
  }

  onStreamingStopped(callback: () => void) {
    this.socket.on('streaming_stopped', callback)
  }

  disconnect() {
    this.socket.disconnect()
  }

  getSocket(): Socket {
    return this.socket
  }
}