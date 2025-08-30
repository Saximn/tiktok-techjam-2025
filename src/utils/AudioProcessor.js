import { Readable, Transform } from 'stream';
import { EventEmitter } from 'events';

/**
 * Real-time Audio Stream Processor
 * Handles live audio streaming with privacy protection
 */
export class AudioStreamProcessor extends Transform {
    constructor(voiceShield, options = {}) {
        super({ objectMode: true });
        
        this.voiceShield = voiceShield;
        this.options = {
            chunkSize: 4096,
            sampleRate: 48000,
            channels: 1,
            bufferSize: 8192,
            ...options
        };
        
        this.audioBuffer = new Float32Array(this.options.bufferSize);
        this.bufferIndex = 0;
        this.isProcessing = false;
        
        this.setupErrorHandling();
    }
    
    /**
     * Transform audio chunks with privacy protection
     * @param {Buffer} chunk 
     * @param {string} encoding 
     * @param {Function} callback 
     */
    async _transform(chunk, encoding, callback) {
        try {
            // Convert Buffer to Float32Array
            const audioData = this.bufferToFloat32Array(chunk);
            
            // Add to processing buffer
            this.addToBuffer(audioData);
            
            // Process if buffer is ready
            if (this.isBufferReady()) {
                const processedAudio = await this.processBufferChunk();
                
                if (processedAudio) {
                    // Convert back to Buffer and push
                    const outputBuffer = this.float32ArrayToBuffer(processedAudio);
                    this.push(outputBuffer);
                }
            }
            
            callback();
        } catch (error) {
            console.error('Audio transform error:', error);
            callback(error);
        }
    }
    
    /**
     * Convert Buffer to Float32Array
     * @param {Buffer} buffer 
     * @returns {Float32Array}
     */
    bufferToFloat32Array(buffer) {
        const samples = new Float32Array(buffer.length / 2);
        
        for (let i = 0; i < samples.length; i++) {
            // Convert 16-bit signed integer to float (-1 to 1)
            const sample = buffer.readInt16LE(i * 2);
            samples[i] = sample / 32768.0;
        }
        
        return samples;
    }
    
    /**
     * Convert Float32Array to Buffer
     * @param {Float32Array} float32Array 
     * @returns {Buffer}
     */
    float32ArrayToBuffer(float32Array) {
        const buffer = Buffer.allocUnsafe(float32Array.length * 2);
        
        for (let i = 0; i < float32Array.length; i++) {
            // Convert float (-1 to 1) to 16-bit signed integer
            const sample = Math.max(-1, Math.min(1, float32Array[i]));
            const intSample = Math.round(sample * 32767);
            buffer.writeInt16LE(intSample, i * 2);
        }
        
        return buffer;
    }
    
    /**
     * Add audio data to processing buffer
     * @param {Float32Array} audioData 
     */
    addToBuffer(audioData) {
        for (let i = 0; i < audioData.length && this.bufferIndex < this.audioBuffer.length; i++) {
            this.audioBuffer[this.bufferIndex++] = audioData[i];
        }
    }
    
    /**
     * Check if buffer is ready for processing
     * @returns {boolean}
     */
    isBufferReady() {
        return this.bufferIndex >= this.options.chunkSize;
    }
    
    /**
     * Process a chunk from the buffer
     * @returns {Promise<Float32Array>}
     */
    async processBufferChunk() {
        if (this.isProcessing) return null;
        
        this.isProcessing = true;
        
        try {
            // Extract chunk from buffer
            const chunk = new Float32Array(this.options.chunkSize);
            chunk.set(this.audioBuffer.slice(0, this.options.chunkSize));
            
            // Shift remaining buffer data
            this.shiftBuffer();
            
            // Process with VoiceShield
            const processedChunk = await this.voiceShield.processRealtimeAudio(chunk);
            
            this.isProcessing = false;
            return processedChunk;
            
        } catch (error) {
            this.isProcessing = false;
            throw error;
        }
    }
    
    /**
     * Shift buffer after processing a chunk
     */
    shiftBuffer() {
        const remaining = this.bufferIndex - this.options.chunkSize;
        
        if (remaining > 0) {
            // Shift remaining data to beginning
            this.audioBuffer.set(
                this.audioBuffer.slice(this.options.chunkSize, this.bufferIndex)
            );
            this.bufferIndex = remaining;
        } else {
            this.bufferIndex = 0;
        }
    }
    
    /**
     * Setup error handling
     */
    setupErrorHandling() {
        this.on('error', (error) => {
            console.error('AudioStreamProcessor error:', error);
            this.emit('processingError', error);
        });
    }
    
    /**
     * Flush remaining buffer on end
     * @param {Function} callback 
     */
    _flush(callback) {
        if (this.bufferIndex > 0) {
            // Process remaining buffer
            this.processBufferChunk()
                .then(processedAudio => {
                    if (processedAudio) {
                        const outputBuffer = this.float32ArrayToBuffer(processedAudio);
                        this.push(outputBuffer);
                    }
                    callback();
                })
                .catch(callback);
        } else {
            callback();
        }
    }
}

/**
 * WebRTC Audio Stream Handler
 * Manages WebRTC connections for real-time audio streaming
 */
export class WebRTCAudioHandler extends EventEmitter {
    constructor(voiceShield) {
        super();
        this.voiceShield = voiceShield;
        this.connections = new Map();
        this.streamProcessors = new Map();
    }
    
    /**
     * Setup WebRTC peer connection for audio streaming
     * @param {string} connectionId 
     * @param {RTCConfiguration} rtcConfig 
     * @returns {RTCPeerConnection}
     */
    setupPeerConnection(connectionId, rtcConfig = {}) {
        const peerConnection = new RTCPeerConnection({
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' }
            ],
            ...rtcConfig
        });
        
        // Handle incoming audio stream
        peerConnection.ontrack = (event) => {
            this.handleIncomingStream(connectionId, event.streams[0]);
        };
        
        // Handle ICE candidates
        peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                this.emit('iceCandidate', {
                    connectionId,
                    candidate: event.candidate
                });
            }
        };
        
        this.connections.set(connectionId, peerConnection);
        return peerConnection;
    }
    
    /**
     * Handle incoming audio stream from WebRTC
     * @param {string} connectionId 
     * @param {MediaStream} stream 
     */
    handleIncomingStream(connectionId, stream) {
        console.log(`📡 Received audio stream from ${connectionId}`);
        
        // Create audio processor for this stream
        const processor = new AudioStreamProcessor(this.voiceShield);
        
        // Setup stream processing pipeline
        this.setupStreamPipeline(connectionId, stream, processor);
        
        this.streamProcessors.set(connectionId, processor);
        this.emit('streamReceived', { connectionId, stream });
    }
    
    /**
     * Setup audio processing pipeline
     * @param {string} connectionId 
     * @param {MediaStream} inputStream 
     * @param {AudioStreamProcessor} processor 
     */
    setupStreamPipeline(connectionId, inputStream, processor) {
        // Create audio context for processing
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(inputStream);
        
        // Create script processor for real-time audio
        const scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        scriptProcessor.onaudioprocess = async (event) => {
            const inputBuffer = event.inputBuffer;
            const outputBuffer = event.outputBuffer;
            
            // Get audio data
            const inputData = inputBuffer.getChannelData(0);
            
            try {
                // Process with VoiceShield
                const protectedData = await this.voiceShield.processRealtimeAudio(inputData);
                
                // Set processed data to output
                const outputData = outputBuffer.getChannelData(0);
                outputData.set(protectedData);
                
            } catch (error) {
                console.error('Real-time processing error:', error);
                // Pass through original audio on error
                const outputData = outputBuffer.getChannelData(0);
                outputData.set(inputData);
            }
        };
        
        // Connect audio pipeline
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        
        // Store pipeline components for cleanup
        this.streamProcessors.set(connectionId, {
            audioContext,
            source,
            scriptProcessor,
            processor
        });
    }
    
    /**
     * Add ICE candidate to peer connection
     * @param {string} connectionId 
     * @param {RTCIceCandidate} candidate 
     */
    async addIceCandidate(connectionId, candidate) {
        const peerConnection = this.connections.get(connectionId);
        if (peerConnection) {
            await peerConnection.addIceCandidate(candidate);
        }
    }
    
    /**
     * Create offer for WebRTC connection
     * @param {string} connectionId 
     * @returns {RTCSessionDescription}
     */
    async createOffer(connectionId) {
        const peerConnection = this.connections.get(connectionId);
        if (!peerConnection) {
            throw new Error(`No connection found for ${connectionId}`);
        }
        
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        return offer;
    }
    
    /**
     * Set remote description for WebRTC connection
     * @param {string} connectionId 
     * @param {RTCSessionDescription} description 
     */
    async setRemoteDescription(connectionId, description) {
        const peerConnection = this.connections.get(connectionId);
        if (peerConnection) {
            await peerConnection.setRemoteDescription(description);
        }
    }
    
    /**
     * Close connection and cleanup resources
     * @param {string} connectionId 
     */
    closeConnection(connectionId) {
        // Close peer connection
        const peerConnection = this.connections.get(connectionId);
        if (peerConnection) {
            peerConnection.close();
            this.connections.delete(connectionId);
        }
        
        // Cleanup stream processor
        const streamProcessor = this.streamProcessors.get(connectionId);
        if (streamProcessor) {
            if (streamProcessor.audioContext) {
                streamProcessor.audioContext.close();
            }
            this.streamProcessors.delete(connectionId);
        }
        
        console.log(`🔌 Closed connection ${connectionId}`);
    }
    
    /**
     * Cleanup all resources
     */
    destroy() {
        // Close all connections
        for (const connectionId of this.connections.keys()) {
            this.closeConnection(connectionId);
        }
        
        this.removeAllListeners();
    }
}

export { AudioStreamProcessor, WebRTCAudioHandler };
