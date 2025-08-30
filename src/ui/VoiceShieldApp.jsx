import React, { useState, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { motion, AnimatePresence } from 'framer-motion';
import { io } from 'socket.io-client';
import './styles.css';

// Modern VoiceShield Web Interface with Latest 2025 Technologies
const VoiceShieldApp = () => {
  // Core state management
  const [isConnected, setIsConnected] = useState(false);
  const [voiceShieldStatus, setVoiceShieldStatus] = useState('offline');
  const [privacyMode, setPrivacyMode] = useState('personal');
  const [protectionLevel, setProtectionLevel] = useState(0.6);
  const [isStreaming, setIsStreaming] = useState(false);
  const [realTimeMetrics, setRealTimeMetrics] = useState({});
  const [socket, setSocket] = useState(null);

  // TikTok Live specific state
  const [streamData, setStreamData] = useState({
    viewerCount: 0,
    streamDuration: 0,
    privacyAlerts: 0,
    piiBlocked: 0,
    backgroundVoicesFiltered: 0
  });

  // Audio visualization state
  const [audioData, setAudioData] = useState(new Float32Array(1024));
  const [spectrogramData, setSpectrogramData] = useState([]);
  const [voiceActivity, setVoiceActivity] = useState(false);
  const [speakers, setSpeakers] = useState([]);

  // Privacy alerts and notifications
  const [notifications, setNotifications] = useState([]);
  const [emergencyMode, setEmergencyMode] = useState(false);

  // Initialize WebSocket connection
  useEffect(() => {
    const newSocket = io('http://localhost:8000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
      setIsConnected(true);
      console.log('Connected to VoiceShield server');
    });

    newSocket.on('disconnect', () => {
      setIsConnected(false);
      console.log('Disconnected from VoiceShield server');
    });

    // Real-time data listeners
    newSocket.on('voiceShieldStatus', (status) => {
      setVoiceShieldStatus(status.mode);
      setProtectionLevel(status.protectionLevel);
    });

    newSocket.on('streamMetrics', (metrics) => {
      setStreamData(prev => ({...prev, ...metrics}));
    });

    newSocket.on('audioData', (data) => {
      setAudioData(new Float32Array(data));
      setVoiceActivity(data.some(val => Math.abs(val) > 0.01));
    });

    newSocket.on('spectrogramUpdate', (data) => {
      setSpectrogramData(prev => [...prev.slice(-49), data]);
    });

    newSocket.on('speakerUpdate', (speakerData) => {
      setSpeakers(speakerData);
    });

    newSocket.on('privacyAlert', (alert) => {
      addNotification(alert);
    });

    return () => {
      newSocket.close();
    };
  }, []);

  // Add notification helper
  const addNotification = useCallback((notification) => {
    const id = Date.now();
    setNotifications(prev => [...prev, { ...notification, id, timestamp: new Date() }]);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  }, []);

  // Handle privacy mode changes
  const handlePrivacyModeChange = useCallback((newMode) => {
    setPrivacyMode(newMode);
    socket?.emit('changePrivacyMode', newMode);
  }, [socket]);

  // Handle protection level changes
  const handleProtectionLevelChange = useCallback((level) => {
    setProtectionLevel(level);
    socket?.emit('adjustProtectionLevel', level);
  }, [socket]);

  // Start/Stop TikTok Live streaming
  const handleStreamingToggle = useCallback(async () => {
    if (isStreaming) {
      socket?.emit('stopLiveStream');
      setIsStreaming(false);
    } else {
      const streamTitle = prompt('Enter stream title:') || 'Live Stream';
      socket?.emit('startLiveStream', { title: streamTitle });
      setIsStreaming(true);
    }
  }, [isStreaming, socket]);

  // Emergency privacy stop
  const handleEmergencyStop = useCallback(() => {
    setEmergencyMode(true);
    socket?.emit('emergencyPrivacyStop');
    addNotification({
      type: 'emergency',
      message: 'Emergency privacy stop activated!',
      severity: 'critical'
    });
  }, [socket, addNotification]);

  return (
    <div className="voiceshield-app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="voice-shield-logo">🛡️</div>
            <div className="title-section">
              <h1>VoiceShield</h1>
              <span className="subtitle">Real-Time AI Voice Privacy Protection</span>
            </div>
          </div>
          
          <div className="connection-status">
            <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
              {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
            </div>
            <div className="mode-indicator">
              Mode: <span className={`mode-${privacyMode}`}>{privacyMode.toUpperCase()}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="dashboard">
        {/* Live Privacy Shield Visualization */}
        <section className="privacy-shield-section">
          <div className="shield-container">
            <div className={`privacy-shield ${voiceActivity ? 'active' : ''} ${emergencyMode ? 'emergency' : ''}`}>
              <div className="shield-core">
                <div className="protection-level">
                  {Math.round(protectionLevel * 100)}%
                </div>
                <div className="protection-label">Protected</div>
              </div>
              
              <div className="shield-rings">
                {[...Array(3)].map((_, i) => (
                  <div
                    key={i}
                    className={`shield-ring ring-${i + 1}`}
                    style={{
                      animationDelay: `${i * 0.5}s`,
                      opacity: protectionLevel * (1 - i * 0.2)
                    }}
                  />
                ))}
              </div>
              
              {voiceActivity && (
                <div className="voice-activity-indicator">
                  <div className="voice-waves">
                    {[...Array(5)].map((_, i) => (
                      <div key={i} className="voice-wave" style={{ animationDelay: `${i * 0.1}s` }} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Control Panel */}
        <section className="control-panel">
          <div className="control-grid">
            {/* Privacy Mode Selector */}
            <div className="control-card">
              <h3>Privacy Mode</h3>
              <div className="privacy-mode-selector">
                {['personal', 'meeting', 'public', 'emergency'].map(mode => (
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

            {/* Protection Level Slider */}
            <div className="control-card">
              <h3>Protection Level</h3>
              <div className="protection-slider-container">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={protectionLevel}
                  onChange={(e) => handleProtectionLevelChange(parseFloat(e.target.value))}
                  className="protection-slider"
                />
                <div className="slider-labels">
                  <span>Low</span>
                  <span>Medium</span>
                  <span>High</span>
                </div>
              </div>
            </div>

            {/* TikTok Live Controls */}
            <div className="control-card tiktok-controls">
              <h3>TikTok Live</h3>
              <div className="live-controls">
                <button
                  className={`live-button ${isStreaming ? 'streaming' : ''}`}
                  onClick={handleStreamingToggle}
                >
                  {isStreaming ? '🔴 Stop Stream' : '▶️ Go Live'}
                </button>
                
                {isStreaming && (
                  <div className="stream-stats">
                    <div className="stat">
                      <span className="stat-value">{streamData.viewerCount}</span>
                      <span className="stat-label">Viewers</span>
                    </div>
                    <div className="stat">
                      <span className="stat-value">{Math.floor(streamData.streamDuration / 60)}m</span>
                      <span className="stat-label">Duration</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Emergency Controls */}
            <div className="control-card emergency-controls">
              <h3>Emergency</h3>
              <button
                className="emergency-button"
                onClick={handleEmergencyStop}
                disabled={emergencyMode}
              >
                {emergencyMode ? '🛑 EMERGENCY ACTIVE' : '🚨 Emergency Stop'}
              </button>
            </div>
          </div>
        </section>

        {/* Real-Time Audio Visualization */}
        <section className="visualization-section">
          <div className="viz-grid">
            {/* Audio Waveform */}
            <div className="viz-card">
              <h3>Live Audio Waveform</h3>
              <div className="waveform-container">
                <svg className="waveform" viewBox="0 0 1000 200">
                  {audioData.map((sample, i) => (
                    <rect
                      key={i}
                      x={i * 2}
                      y={100 - sample * 100}
                      width="1.5"
                      height={Math.abs(sample) * 200}
                      fill={`hsl(${120 - Math.abs(sample) * 120}, 70%, 50%)`}
                      opacity={0.8}
                    />
                  )).slice(0, 500)}
                </svg>
              </div>
            </div>

            {/* Voice Spectrogram */}
            <div className="viz-card">
              <h3>Voice Spectrogram</h3>
              <div className="spectrogram-container">
                <div className="spectrogram">
                  {spectrogramData.map((frame, frameIndex) => (
                    <div key={frameIndex} className="spectrogram-column">
                      {frame && frame.map((bin, binIndex) => (
                        <div
                          key={binIndex}
                          className="spectrogram-bin"
                          style={{
                            backgroundColor: `hsl(${240 - bin * 240}, 70%, ${30 + bin * 50}%)`,
                            opacity: bin
                          }}
                        />
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Speaker Map */}
            <div className="viz-card">
              <h3>Speaker Detection</h3>
              <div className="speaker-map">
                {speakers.length === 0 ? (
                  <div className="no-speakers">No speakers detected</div>
                ) : (
                  speakers.map((speaker, i) => (
                    <div key={speaker.id || i} className="speaker-indicator">
                      <div className={`speaker-avatar ${speaker.isMainSpeaker ? 'main' : ''}`}>
                        {speaker.isMainSpeaker ? '👤' : '👥'}
                      </div>
                      <div className="speaker-info">
                        <div className="speaker-label">
                          {speaker.isMainSpeaker ? 'Main Speaker' : `Speaker ${i + 1}`}
                        </div>
                        <div className="speaker-confidence">
                          {Math.round(speaker.confidence * 100)}% confident
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Privacy Metrics Dashboard */}
        <section className="metrics-section">
          <h2>Privacy Protection Metrics</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-value">{streamData.privacyAlerts}</div>
              <div className="metric-label">Privacy Alerts</div>
              <div className="metric-change">Today</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-value">{streamData.piiBlocked}</div>
              <div className="metric-label">PII Blocked</div>
              <div className="metric-change">This session</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-value">{streamData.backgroundVoicesFiltered}</div>
              <div className="metric-label">Background Voices Filtered</div>
              <div className="metric-change">This session</div>
            </div>
            
            <div className="metric-card">
              <div className="metric-value">{Math.round(protectionLevel * 100)}%</div>
              <div className="metric-label">Current Protection</div>
              <div className="metric-change">Real-time</div>
            </div>
          </div>
        </section>
      </main>

      {/* Notifications */}
      <AnimatePresence>
        {notifications.map(notification => (
          <motion.div
            key={notification.id}
            className={`notification ${notification.type} ${notification.severity || 'info'}`}
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            transition={{ type: "spring", stiffness: 100, damping: 15 }}
          >
            <div className="notification-content">
              <div className="notification-message">{notification.message}</div>
              <div className="notification-time">
                {notification.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};

// Initialize React App
const container = document.getElementById('root');
const root = createRoot(container);
root.render(<VoiceShieldApp />);

export default VoiceShieldApp;
