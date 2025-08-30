import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import VoiceShield from './VoiceShield.jsx';

/**
 * TikTok Live VoiceShield Component
 * Specialized interface for TikTok Live streaming with privacy protection
 */
const TikTokLiveVoiceShield = ({ userId, serverUrl = 'http://localhost:3001' }) => {
    // TikTok Live specific state
    const [isLiveStreaming, setIsLiveStreaming] = useState(false);
    const [viewerCount, setViewerCount] = useState(0);
    const [isEmergencyMuted, setIsEmergencyMuted] = useState(false);
    const [streamMetrics, setStreamMetrics] = useState({});
    const [audienceRisk, setAudienceRisk] = useState('low');
    const [backgroundProtection, setBackgroundProtection] = useState(true);
    const [isInitialized, setIsInitialized] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);
    
    // Refs
    const socketRef = useRef(null);
    const streamTimerRef = useRef(null);
    const streamStartTime = useRef(null);
    
    useEffect(() => {
        initializeTikTokLive();
        
        return () => {
            cleanup();
        };
    }, [userId]);
    
    /**
     * Initialize TikTok Live VoiceShield
     */
    const initializeTikTokLive = async () => {
        try {
            setError(null);
            
            // Initialize TikTok Live backend
            const response = await fetch(`${serverUrl}/api/tiktok/init`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userId,
                    streamConfig: {
                        ultraLowLatency: true,
                        targetLatency: 25,
                        audienceAwareProtection: true,
                        backgroundProtection: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to initialize TikTok Live VoiceShield');
            }
            
            // Setup WebSocket for live features
            await setupTikTokWebSocket();
            
            setIsInitialized(true);
            console.log('✅ TikTok Live VoiceShield initialized');
            
        } catch (error) {
            console.error('❌ TikTok Live initialization failed:', error);
            setError(error.message);
        }
    };
    
    /**
     * Setup WebSocket connection for TikTok Live features
     */
    const setupTikTokWebSocket = () => {
        return new Promise((resolve, reject) => {
            socketRef.current = io(serverUrl);
            
            socketRef.current.on('connect', () => {
                setIsConnected(true);
                socketRef.current.emit('join-session', { userId });
            });
            
            socketRef.current.on('session-joined', (data) => {
                console.log('📺 Joined TikTok Live session:', data);
                resolve();
            });
            
            socketRef.current.on('metrics-update', (data) => {
                setStreamMetrics(data.tikTokLive || {});
                setAudienceRisk(data.tikTokLive?.audienceRisk || 'low');
            });
            
            socketRef.current.on('viewer-count-updated', (data) => {
                setViewerCount(data.viewerCount);
            });
            
            socketRef.current.on('emergency-mute-activated', () => {
                setIsEmergencyMuted(true);
            });
            
            socketRef.current.on('emergency-mute-deactivated', () => {
                setIsEmergencyMuted(false);
            });
            
            socketRef.current.on('error', (error) => {
                console.error('TikTok WebSocket error:', error);
                setError(error.message);
            });
            
            setTimeout(() => {
                if (!socketRef.current?.connected) {
                    reject(new Error('TikTok WebSocket connection timeout'));
                }
            }, 10000);
        });
    };
    
    /**
     * Start TikTok Live stream with privacy protection
     */
    const startLiveStream = async () => {
        try {
            const response = await fetch(`${serverUrl}/api/tiktok/start-stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userId,
                    streamConfig: {
                        initialViewers: viewerCount,
                        backgroundProtection,
                        audienceAware: true
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to start TikTok Live stream');
            }
            
            setIsLiveStreaming(true);
            streamStartTime.current = Date.now();
            
            // Start stream timer
            streamTimerRef.current = setInterval(() => {
                // Update stream duration display
            }, 1000);
            
            console.log('🔴 TikTok Live stream started with privacy protection');
            
        } catch (error) {
            console.error('❌ Failed to start live stream:', error);
            setError(error.message);
        }
    };
    
    /**
     * Stop TikTok Live stream
     */
    const stopLiveStream = async () => {
        try {
            const response = await fetch(`${serverUrl}/api/tiktok/stop-stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userId })
            });
            
            if (!response.ok) {
                throw new Error('Failed to stop TikTok Live stream');
            }
            
            setIsLiveStreaming(false);
            
            // Clear stream timer
            if (streamTimerRef.current) {
                clearInterval(streamTimerRef.current);
                streamTimerRef.current = null;
            }
            
            console.log('⏹️ TikTok Live stream stopped');
            
        } catch (error) {
            console.error('❌ Failed to stop live stream:', error);
            setError(error.message);
        }
    };
    
    /**
     * Emergency mute functionality
     */
    const handleEmergencyMute = () => {
        if (socketRef.current?.connected) {
            socketRef.current.emit('emergency-mute');
        }
    };
    
    /**
     * Emergency unmute functionality
     */
    const handleEmergencyUnmute = () => {
        if (socketRef.current?.connected) {
            socketRef.current.emit('emergency-unmute');
        }
    };
    
    /**
     * Simulate viewer count change (in real app, this would come from TikTok API)
     */
    const simulateViewerCountChange = (newCount) => {
        setViewerCount(newCount);
        if (socketRef.current?.connected) {
            socketRef.current.emit('viewer-count-update', { viewerCount: newCount });
        }
    };
    
    /**
     * Get stream duration
     */
    const getStreamDuration = () => {
        if (!streamStartTime.current) return '00:00';
        
        const duration = Math.floor((Date.now() - streamStartTime.current) / 1000);
        const minutes = Math.floor(duration / 60);
        const seconds = duration % 60;
        
        return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    };
    
    /**
     * Get audience risk color
     */
    const getAudienceRiskColor = () => {
        switch (audienceRisk) {
            case 'high': return '#ff4757';
            case 'medium': return '#ffa502';
            case 'low': return '#2ed573';
            default: return '#747d8c';
        }
    };
    
    /**
     * Cleanup resources
     */
    const cleanup = () => {
        if (isLiveStreaming) {
            stopLiveStream();
        }
        
        if (streamTimerRef.current) {
            clearInterval(streamTimerRef.current);
        }
        
        if (socketRef.current) {
            socketRef.current.disconnect();
        }
    };
    
    if (error) {
        return (
            <div className="tiktok-live-container error">
                <div className="error-message">
                    <h3>❌ TikTok Live Error</h3>
                    <p>{error}</p>
                    <button onClick={initializeTikTokLive} className="retry-button">
                        Retry
                    </button>
                </div>
            </div>
        );
    }
    
    return (
        <div className="tiktok-live-container">
            <header className="tiktok-live-header">
                <div className="header-left">
                    <h1>📺 TikTok Live VoiceShield</h1>
                    {isLiveStreaming && (
                        <div className="live-indicator">
                            <span className="live-badge">🔴 LIVE</span>
                            <span className="stream-duration">{getStreamDuration()}</span>
                        </div>
                    )}
                </div>
                <div className="header-right">
                    <div className="viewer-count">
                        <span className="viewer-icon">👁️</span>
                        <span className="count">{viewerCount.toLocaleString()}</span>
                    </div>
                </div>
            </header>
            
            <main className="tiktok-live-main">
                {/* Live Stream Controls */}
                <section className="stream-controls">
                    <div className="primary-controls">
                        {!isLiveStreaming ? (
                            <button
                                onClick={startLiveStream}
                                className="stream-button start"
                                disabled={!isInitialized || !isConnected}
                            >
                                🔴 Go Live with Privacy Protection
                            </button>
                        ) : (
                            <button
                                onClick={stopLiveStream}
                                className="stream-button stop"
                            >
                                ⏹️ End Live Stream
                            </button>
                        )}
                        
                        {/* Emergency Mute Button */}
                        {isLiveStreaming && (
                            <div className="emergency-controls">
                                {!isEmergencyMuted ? (
                                    <button
                                        onClick={handleEmergencyMute}
                                        className="emergency-button mute"
                                        title="Emergency Privacy Mute"
                                    >
                                        🚨 Emergency Mute
                                    </button>
                                ) : (
                                    <button
                                        onClick={handleEmergencyUnmute}
                                        className="emergency-button unmute"
                                        title="Deactivate Emergency Mute"
                                    >
                                        🔊 Unmute
                                    </button>
                                )}
                            </div>
                        )}
                    </div>
                </section>
                
                {/* Audience Analytics */}
                <section className="audience-analytics">
                    <h3>👥 Audience Privacy Analytics</h3>
                    <div className="analytics-grid">
                        <div className="analytics-card">
                            <div className="card-header">
                                <span className="card-icon">📊</span>
                                <span className="card-title">Viewer Count</span>
                            </div>
                            <div className="card-value">{viewerCount.toLocaleString()}</div>
                            <div className="card-controls">
                                <button onClick={() => simulateViewerCountChange(viewerCount + 10)}>
                                    +10
                                </button>
                                <button onClick={() => simulateViewerCountChange(viewerCount + 100)}>
                                    +100
                                </button>
                                <button onClick={() => simulateViewerCountChange(viewerCount + 1000)}>
                                    +1K
                                </button>
                            </div>
                        </div>
                        
                        <div className="analytics-card">
                            <div className="card-header">
                                <span className="card-icon">⚠️</span>
                                <span className="card-title">Audience Risk</span>
                            </div>
                            <div 
                                className="card-value risk-level"
                                style={{ color: getAudienceRiskColor() }}
                            >
                                {audienceRisk.toUpperCase()}
                            </div>
                            <div className="risk-description">
                                {audienceRisk === 'high' && 'Maximum privacy protection active'}
                                {audienceRisk === 'medium' && 'Balanced privacy protection'}
                                {audienceRisk === 'low' && 'Standard privacy protection'}
                            </div>
                        </div>
                        
                        <div className="analytics-card">
                            <div className="card-header">
                                <span className="card-icon">📱</span>
                                <span className="card-title">Stream Health</span>
                            </div>
                            <div className="card-value">
                                {streamMetrics.averageLatency ? `${streamMetrics.averageLatency}ms` : 'N/A'}
                            </div>
                            <div className="health-indicators">
                                <div className={`indicator ${streamMetrics.averageLatency < 30 ? 'good' : 'warning'}`}>
                                    Latency
                                </div>
                                <div className={`indicator ${!streamMetrics.privacyBreaches ? 'good' : 'warning'}`}>
                                    Privacy
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
                
                {/* Live Privacy Settings */}
                <section className="live-privacy-settings">
                    <h3>🔒 Live Privacy Settings</h3>
                    <div className="settings-grid">
                        <div className="setting-item">
                            <label className="setting-label">
                                <input
                                    type="checkbox"
                                    checked={backgroundProtection}
                                    onChange={(e) => setBackgroundProtection(e.target.checked)}
                                />
                                Background Voice Protection
                            </label>
                            <p className="setting-description">
                                Automatically filter out family members and background conversations
                            </p>
                        </div>
                        
                        <div className="setting-item">
                            <label className="setting-label">
                                Location Masking
                            </label>
                            <p className="setting-description">
                                Prevent accidental location reveals in background audio
                            </p>
                        </div>
                        
                        <div className="setting-item">
                            <label className="setting-label">
                                PII Auto-Detection
                            </label>
                            <p className="setting-description">
                                Automatically detect and mask personal information
                            </p>
                        </div>
                    </div>
                </section>
                
                {/* Emergency Status */}
                {isEmergencyMuted && (
                    <section className="emergency-status">
                        <div className="emergency-alert">
                            <div className="alert-icon">🚨</div>
                            <div className="alert-content">
                                <h3>Emergency Privacy Mode Active</h3>
                                <p>
                                    Audio is currently muted for privacy protection. 
                                    Viewers see "Technical Difficulties" message.
                                </p>
                            </div>
                        </div>
                    </section>
                )}
                
                {/* Embed VoiceShield Component for core functionality */}
                <VoiceShield userId={userId} serverUrl={serverUrl} />
            </main>
            
            <footer className="tiktok-live-footer">
                <div className="footer-stats">
                    <span>Stream Quality: {streamMetrics.averageLatency < 30 ? '🟢 Excellent' : '🟡 Good'}</span>
                    <span>•</span>
                    <span>Privacy Breaches: {streamMetrics.privacyBreaches || 0}</span>
                    <span>•</span>
                    <span>Protection Level: {Math.round((streamMetrics.coreMetrics?.privacyScore || 0.7) * 100)}%</span>
                </div>
            </footer>
        </div>
    );
};

export default TikTokLiveVoiceShield;
