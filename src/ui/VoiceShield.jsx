import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';

/**
 * VoiceShield React Component
 * Real-time voice privacy protection interface
 */
const VoiceShield = ({ userId, serverUrl = 'http://localhost:3001' }) => {
    // State management
    const [isInitialized, setIsInitialized] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    const [privacyMode, setPrivacyMode] = useState('balanced');
    const [isRecording, setIsRecording] = useState(false);
    const [metrics, setMetrics] = useState({});
    const [privacyLevel, setPrivacyLevel] = useState(0.7);
    const [piiDetected, setPiiDetected] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    
    // Refs
    const socketRef = useRef(null);
    const audioContextRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const streamRef = useRef(null);
    const canvasRef = useRef(null);
    const animationRef = useRef(null);
    
    // Initialize VoiceShield
    useEffect(() => {
        initializeVoiceShield();
        
        return () => {
            cleanup();
        };
    }, [userId]);
    
    /**
     * Initialize VoiceShield session
     */
    const initializeVoiceShield = async () => {
        try {
            setIsLoading(true);
            setError(null);
            
            // Initialize VoiceShield backend
            const response = await fetch(`${serverUrl}/api/voiceshield/init`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userId,
                    config: {
                        privacyMode: 'balanced',
                        targetLatency: 50,
                        enablePIIDetection: true,
                        enableEmotionMasking: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to initialize VoiceShield');
            }
            
            const data = await response.json();
            console.log('✅ VoiceShield initialized:', data);
            
            // Setup WebSocket connection
            await setupWebSocket();
            
            setIsInitialized(true);
            setIsLoading(false);
            
        } catch (error) {
            console.error('❌ VoiceShield initialization failed:', error);
            setError(error.message);
            setIsLoading(false);
        }
    };
    
    /**
     * Setup WebSocket connection for real-time communication
     */
    const setupWebSocket = () => {
        return new Promise((resolve, reject) => {
            socketRef.current = io(serverUrl);
            
            socketRef.current.on('connect', () => {
                console.log('🔌 Connected to VoiceShield server');
                setIsConnected(true);
                
                // Join user session
                socketRef.current.emit('join-session', { userId });
            });
            
            socketRef.current.on('session-joined', (data) => {
                console.log('👤 Joined session:', data);
                resolve();
            });
            
            socketRef.current.on('processed-audio', (data) => {
                // Handle processed audio data
                handleProcessedAudio(data);
            });
            
            socketRef.current.on('metrics-update', (data) => {
                setMetrics(data);
            });
            
            socketRef.current.on('privacy-mode-updated', (data) => {
                setPrivacyMode(data.mode);
            });
            
            socketRef.current.on('error', (error) => {
                console.error('WebSocket error:', error);
                setError(error.message);
            });
            
            socketRef.current.on('disconnect', () => {
                console.log('🔌 Disconnected from server');
                setIsConnected(false);
            });
            
            // Timeout for connection
            setTimeout(() => {
                if (!socketRef.current?.connected) {
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 10000);
        });
    };
    
    /**
     * Start audio recording and processing
     */
    const startRecording = async () => {
        try {
            if (!isInitialized || !isConnected) {
                throw new Error('VoiceShield not ready');
            }
            
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 48000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            streamRef.current = stream;
            
            // Setup audio context for processing
            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContextRef.current.createMediaStreamSource(stream);
            
            // Create analyzer for visualization
            const analyzer = audioContextRef.current.createAnalyser();
            analyzer.fftSize = 2048;
            source.connect(analyzer);
            
            // Start visualization
            startVisualization(analyzer);
            
            // Setup audio processing
            setupAudioProcessing(stream);
            
            setIsRecording(true);
            console.log('🎤 Recording started with privacy protection');
            
        } catch (error) {
            console.error('❌ Failed to start recording:', error);
            setError(error.message);
        }
    };
    
    /**
     * Stop audio recording
     */
    const stopRecording = () => {
        try {
            // Stop media recorder
            if (mediaRecorderRef.current?.state === 'recording') {
                mediaRecorderRef.current.stop();
            }
            
            // Stop audio stream
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
                streamRef.current = null;
            }
            
            // Close audio context
            if (audioContextRef.current?.state !== 'closed') {
                audioContextRef.current.close();
                audioContextRef.current = null;
            }
            
            // Stop visualization
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
                animationRef.current = null;
            }
            
            setIsRecording(false);
            console.log('🎤 Recording stopped');
            
        } catch (error) {
            console.error('❌ Failed to stop recording:', error);
        }
    };
    
    /**
     * Setup audio processing pipeline
     */
    const setupAudioProcessing = (stream) => {
        const mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=pcm'
        });
        
        mediaRecorderRef.current = mediaRecorder;
        
        // Process audio in chunks
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0 && socketRef.current?.connected) {
                // Convert to base64 and send to server
                const reader = new FileReader();
                reader.onload = () => {
                    const audioData = reader.result.split(',')[1];
                    socketRef.current.emit('audio-chunk', {
                        audioData,
                        chunkId: Date.now(),
                        timestamp: Date.now()
                    });
                };
                reader.readAsDataURL(event.data);
            }
        };
        
        // Start recording with 100ms chunks for real-time processing
        mediaRecorder.start(100);
    };
    
    /**
     * Handle processed audio from server
     */
    const handleProcessedAudio = (data) => {
        // In a real implementation, you might play the processed audio
        // or update UI based on privacy analysis
        console.log('🔒 Processed audio received:', {
            chunkId: data.chunkId,
            latency: Date.now() - data.originalTimestamp
        });
        
        // Request updated metrics
        socketRef.current?.emit('get-metrics');
    };
    
    /**
     * Start audio visualization
     */
    const startVisualization = (analyzer) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const bufferLength = analyzer.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            animationRef.current = requestAnimationFrame(draw);
            
            analyzer.getByteFrequencyData(dataArray);
            
            ctx.fillStyle = 'rgb(15, 15, 35)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i] / 2;
                
                // Color based on privacy level
                const r = Math.floor(255 * (1 - privacyLevel));
                const g = Math.floor(255 * privacyLevel);
                const b = 100;
                
                ctx.fillStyle = `rgb(${r},${g},${b})`;
                ctx.fillRect(x, canvas.height - barHeight / 2, barWidth, barHeight);
                
                x += barWidth + 1;
            }
            
            // Draw privacy indicator
            drawPrivacyIndicator(ctx, canvas);
        };
        
        draw();
    };
    
    /**
     * Draw privacy protection indicator
     */
    const drawPrivacyIndicator = (ctx, canvas) => {
        const centerX = canvas.width - 50;
        const centerY = 50;
        const radius = 20;
        
        // Draw privacy shield
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        
        // Color based on privacy level
        const hue = privacyLevel * 120; // Green to red
        ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
        ctx.fill();
        
        // Draw shield icon
        ctx.fillStyle = 'white';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('🛡️', centerX, centerY + 5);
        
        // Privacy level text
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(`${Math.round(privacyLevel * 100)}%`, centerX, centerY + 35);
    };
}
    
    /**
     * Handle privacy mode change
     */
    const handlePrivacyModeChange = (newMode) => {
        if (socketRef.current?.connected) {
            socketRef.current.emit('set-privacy-mode', { mode: newMode });
        }
    };
    
    /**
     * Handle privacy level slider change
     */
    const handlePrivacyLevelChange = (level) => {
        setPrivacyLevel(level);
        
        // Map slider to privacy mode
        let mode = 'balanced';
        if (level < 0.4) mode = 'minimal';
        else if (level > 0.8) mode = 'maximum';
        
        handlePrivacyModeChange(mode);
    };
    
    /**
     * Cleanup resources
     */
    const cleanup = () => {
        stopRecording();
        
        if (socketRef.current) {
            socketRef.current.disconnect();
        }
    };
    
    // Loading state
    if (isLoading) {
        return (
            <div className="voiceshield-container loading">
                <div className="loading-spinner"></div>
                <p>Initializing VoiceShield AI models...</p>
            </div>
        );
    }
    
    // Error state
    if (error) {
        return (
            <div className="voiceshield-container error">
                <div className="error-message">
                    <h3>❌ VoiceShield Error</h3>
                    <p>{error}</p>
                    <button onClick={initializeVoiceShield} className="retry-button">
                        Retry
                    </button>
                </div>
            </div>
        );
    }
    
    return (
        <div className="voiceshield-container">
            <header className="voiceshield-header">
                <h1>🛡️ VoiceShield</h1>
                <div className="status-indicators">
                    <div className={`status-indicator ${isInitialized ? 'active' : 'inactive'}`}>
                        <span className="indicator-dot"></span>
                        AI Models
                    </div>
                    <div className={`status-indicator ${isConnected ? 'active' : 'inactive'}`}>
                        <span className="indicator-dot"></span>
                        Server
                    </div>
                    <div className={`status-indicator ${isRecording ? 'active' : 'inactive'}`}>
                        <span className="indicator-dot"></span>
                        Recording
                    </div>
                </div>
            </header>
            
            <main className="voiceshield-main">
                {/* Audio Visualization */}
                <section className="audio-visualization">
                    <canvas 
                        ref={canvasRef}
                        width={600}
                        height={200}
                        className="audio-canvas"
                    />
                    {isRecording && (
                        <div className="recording-indicator">
                            <div className="pulse-ring"></div>
                            <div className="recording-dot"></div>
                        </div>
                    )}
                </section>
                
                {/* Privacy Controls */}
                <section className="privacy-controls">
                    <div className="privacy-level-control">
                        <label htmlFor="privacy-slider">Privacy Protection Level</label>
                        <div className="slider-container">
                            <span className="slider-label">Minimal</span>
                            <input
                                id="privacy-slider"
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={privacyLevel}
                                onChange={(e) => handlePrivacyLevelChange(parseFloat(e.target.value))}
                                className="privacy-slider"
                            />
                            <span className="slider-label">Maximum</span>
                        </div>
                        <div className="privacy-percentage">
                            {Math.round(privacyLevel * 100)}% Protected
                        </div>
                    </div>
                    
                    <div className="privacy-modes">
                        <h3>Quick Privacy Modes</h3>
                        <div className="mode-buttons">
                            {['minimal', 'balanced', 'maximum'].map(mode => (
                                <button
                                    key={mode}
                                    className={`mode-button ${privacyMode === mode ? 'active' : ''}`}
                                    onClick={() => handlePrivacyModeChange(mode)}
                                >
                                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                                </button>
                            ))}
                        </div>
                    </div>
                </section>
                
                {/* Recording Controls */}
                <section className="recording-controls">
                    {!isRecording ? (
                        <button
                            onClick={startRecording}
                            className="record-button start"
                            disabled={!isInitialized || !isConnected}
                        >
                            🎤 Start Privacy Protection
                        </button>
                    ) : (
                        <button
                            onClick={stopRecording}
                            className="record-button stop"
                        >
                            ⏹️ Stop Recording
                        </button>
                    )}
                </section>
                
                {/* Privacy Metrics */}
                <section className="privacy-metrics">
                    <h3>Real-time Privacy Metrics</h3>
                    <div className="metrics-grid">
                        <div className="metric-card">
                            <div className="metric-value">
                                {metrics.voiceShield?.latency || 0}ms
                            </div>
                            <div className="metric-label">Processing Latency</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value">
                                {Math.round((metrics.voiceShield?.privacyScore || 0) * 100)}%
                            </div>
                            <div className="metric-label">Privacy Score</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value">
                                {metrics.voiceShield?.piiDetected || 0}
                            </div>
                            <div className="metric-label">PII Detected</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-value">
                                {metrics.server?.activeConnections || 0}
                            </div>
                            <div className="metric-label">Active Users</div>
                        </div>
                    </div>
                </section>
                
                {/* PII Detection Alerts */}
                {piiDetected.length > 0 && (
                    <section className="pii-alerts">
                        <h3>⚠️ Privacy Alerts</h3>
                        <div className="alert-list">
                            {piiDetected.map((pii, index) => (
                                <div key={index} className="alert-item">
                                    <span className="alert-type">{pii.type}</span>
                                    <span className="alert-text">{pii.text}</span>
                                    <span className="alert-confidence">
                                        {Math.round(pii.confidence * 100)}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    </section>
                )}
                
                {/* Privacy Suggestions */}
                {metrics.voiceShield?.suggestions?.length > 0 && (
                    <section className="privacy-suggestions">
                        <h3>💡 Privacy Suggestions</h3>
                        <ul className="suggestion-list">
                            {metrics.voiceShield.suggestions.map((suggestion, index) => (
                                <li key={index} className="suggestion-item">
                                    {suggestion}
                                </li>
                            ))}
                        </ul>
                    </section>
                )}
            </main>
            
            <footer className="voiceshield-footer">
                <div className="footer-info">
                    <span>VoiceShield v1.0.0</span>
                    <span>•</span>
                    <span>Real-time AI Voice Privacy Protection</span>
                </div>
            </footer>
        </div>
    );
};

export default VoiceShield;
