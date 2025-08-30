import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Mock Lynx UI Components (since Lynx packages would need to be installed)
const LynxCore = {
  AudioVisualizer: ({ data, className }) => (
    <div className={`audio-visualizer ${className}`}>
      <svg width="100%" height="60">
        {data.map((value, index) => (
          <rect
            key={index}
            x={index * 4}
            y={30 - value * 30}
            width="3"
            height={value * 60}
            fill={`hsl(${120 - value * 120}, 70%, 50%)`}
          />
        ))}
      </svg>
    </div>
  ),
  
  PrivacyShield: ({ level, active, children }) => (
    <div className={`privacy-shield ${active ? 'active' : ''}`} style={{
      '--privacy-level': level,
      background: `conic-gradient(from 0deg, 
        hsl(${120 * level}, 70%, 50%) 0deg, 
        hsl(${120 * level}, 70%, 50%) ${360 * level}deg, 
        rgba(255,255,255,0.1) ${360 * level}deg)`
    }}>
      {children}
    </div>
  ),

  Button: ({ variant, onClick, children, disabled }) => (
    <button 
      className={`lynx-button ${variant}`} 
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  )
};

const VoiceShieldDashboard = () => {
  const [isActive, setIsActive] = useState(false);
  const [privacyLevel, setPrivacyLevel] = useState(0.6);
  const [privacyMode, setPrivacyMode] = useState('personal');
  const [audioData, setAudioData] = useState(Array.from({length: 50}, () => Math.random()));
  const [metrics, setMetrics] = useState({
    latency: 0,
    protection: 0,
    piiDetected: 0,
    voicesMasked: 0
  });
  const [alerts, setAlerts] = useState([]);
  const [streamingMode, setStreamingMode] = useState(false);
  const [viewerCount, setViewerCount] = useState(0);

  // Simulate real-time audio data
  useEffect(() => {
    if (isActive) {
      const interval = setInterval(() => {
        setAudioData(prev => {
          const newData = [...prev.slice(1), Math.random() * (streamingMode ? 0.8 : 0.5)];
          return newData;
        });
        
        // Simulate processing metrics
        setMetrics(prev => ({
          latency: 15 + Math.random() * 35, // 15-50ms
          protection: privacyLevel * 100,
          piiDetected: Math.random() < 0.1 ? prev.piiDetected + 1 : prev.piiDetected,
          voicesMasked: Math.random() < 0.2 ? prev.voicesMasked + 1 : prev.voicesMasked
        }));
        
        // Simulate alerts
        if (Math.random() < 0.05) {
          const alertTypes = ['PII Detected', 'Background Voice', 'High Risk'];
          const newAlert = {
            id: Date.now(),
            type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
            timestamp: new Date().toLocaleTimeString()
          };
          setAlerts(prev => [newAlert, ...prev.slice(0, 4)]);
        }
      }, 100);

      return () => clearInterval(interval);
    }
  }, [isActive, streamingMode, privacyLevel]);

  // TikTok Live simulation
  useEffect(() => {
    if (streamingMode) {
      const interval = setInterval(() => {
        setViewerCount(prev => Math.max(0, prev + Math.floor(Math.random() * 20 - 10)));
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [streamingMode]);

  const handleEmergencyStop = () => {
    setIsActive(false);
    setStreamingMode(false);
    setAlerts(prev => [{
      id: Date.now(),
      type: 'EMERGENCY STOP',
      timestamp: new Date().toLocaleTimeString()
    }, ...prev.slice(0, 4)]);
  };

  const handleStartTikTokLive = () => {
    setStreamingMode(true);
    setIsActive(true);
    setViewerCount(Math.floor(Math.random() * 100 + 10));
    setPrivacyMode('streaming');
  };

  return (
    <div className="voiceshield-dashboard">
      <header className="dashboard-header">
        <h1>🛡️ VoiceShield</h1>
        <p>Real-Time AI Voice Privacy Protection - TikTok TechJam 2025</p>
      </header>

      <div className="dashboard-grid">
        {/* Main Privacy Shield */}
        <div className="panel main-shield">
          <h2>Privacy Protection</h2>
          <LynxCore.PrivacyShield level={privacyLevel} active={isActive}>
            <div className="shield-center">
              <div className="protection-level">{(privacyLevel * 100).toFixed(0)}%</div>
              <div className="status">{isActive ? 'PROTECTED' : 'INACTIVE'}</div>
            </div>
          </LynxCore.PrivacyShield>
          
          <div className="privacy-controls">
            <label>Privacy Level</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={privacyLevel}
              onChange={(e) => setPrivacyLevel(parseFloat(e.target.value))}
              className="privacy-slider"
            />
            
            <div className="mode-buttons">
              {['personal', 'meeting', 'public', 'streaming'].map(mode => (
                <button
                  key={mode}
                  className={`mode-btn ${privacyMode === mode ? 'active' : ''}`}
                  onClick={() => {
                    setPrivacyMode(mode);
                    setPrivacyLevel(mode === 'personal' ? 0.6 : mode === 'meeting' ? 0.8 : mode === 'public' ? 1.0 : 0.85);
                  }}
                >
                  {mode.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Real-time Audio Visualization */}
        <div className="panel audio-viz">
          <h3>Live Audio Stream</h3>
          <LynxCore.AudioVisualizer data={audioData} className="main-visualizer" />
          <div className="audio-info">
            <span>Sample Rate: 48kHz</span>
            <span>Chunk Size: 20ms</span>
            <span>Status: {isActive ? '🟢 LIVE' : '🔴 IDLE'}</span>
          </div>
        </div>

        {/* TikTok Live Integration */}
        <div className="panel tiktok-live">
          <h3>📱 TikTok Live</h3>
          {streamingMode ? (
            <div className="live-stats">
              <div className="viewer-count">👥 {viewerCount} viewers</div>
              <div className="stream-status">🔴 LIVE</div>
              <div className="stream-metrics">
                <div>Latency: {metrics.latency.toFixed(1)}ms</div>
                <div>Protection: {metrics.protection.toFixed(0)}%</div>
              </div>
              <LynxCore.Button variant="danger" onClick={() => setStreamingMode(false)}>
                End Stream
              </LynxCore.Button>
            </div>
          ) : (
            <div className="stream-controls">
              <p>Ready to go live with privacy protection</p>
              <LynxCore.Button variant="primary" onClick={handleStartTikTokLive}>
                🚀 Start TikTok Live
              </LynxCore.Button>
            </div>
          )}
        </div>

        {/* Performance Metrics */}
        <div className="panel metrics">
          <h3>⚡ Performance</h3>
          <div className="metrics-grid">
            <div className="metric">
              <div className="metric-value">{metrics.latency.toFixed(1)}ms</div>
              <div className="metric-label">Latency</div>
            </div>
            <div className="metric">
              <div className="metric-value">{metrics.protection.toFixed(0)}%</div>
              <div className="metric-label">Protection</div>
            </div>
            <div className="metric">
              <div className="metric-value">{metrics.piiDetected}</div>
              <div className="metric-label">PII Blocked</div>
            </div>
            <div className="metric">
              <div className="metric-value">{metrics.voicesMasked}</div>
              <div className="metric-label">Voices Masked</div>
            </div>
          </div>
        </div>

        {/* Privacy Alerts */}
        <div className="panel alerts">
          <h3>🚨 Privacy Alerts</h3>
          <div className="alerts-list">
            {alerts.length === 0 ? (
              <div className="no-alerts">No privacy threats detected</div>
            ) : (
              alerts.map(alert => (
                <div key={alert.id} className="alert-item">
                  <span className="alert-type">{alert.type}</span>
                  <span className="alert-time">{alert.timestamp}</span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* SOTA Models Status */}
        <div className="panel models-status">
          <h3>🤖 SOTA Models</h3>
          <div className="models-list">
            {[
              'Whisper-v3 (Speech Recognition)',
              'StyleTTS2 (Voice Conversion)', 
              'Pyannote 3.0 (Speaker Diarization)',
              'WavLM (Emotion Detection)',
              'AudioCraft (Audio Inpainting)'
            ].map((model, index) => (
              <div key={index} className="model-status">
                <span className="model-name">{model}</span>
                <span className="model-indicator">🟢</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div className="control-panel">
        <LynxCore.Button 
          variant={isActive ? "secondary" : "primary"}
          onClick={() => setIsActive(!isActive)}
        >
          {isActive ? '⏸️ Pause Protection' : '▶️ Start Protection'}
        </LynxCore.Button>
        
        <LynxCore.Button 
          variant="danger"
          onClick={handleEmergencyStop}
          disabled={!isActive}
        >
          🚨 Emergency Stop
        </LynxCore.Button>
        
        <div className="status-indicator">
          <div className={`indicator ${isActive ? 'active' : 'inactive'}`}></div>
          <span>VoiceShield {isActive ? 'Active' : 'Standby'}</span>
        </div>
      </div>

      <footer className="dashboard-footer">
        <p>🛡️ VoiceShield v2.0 - TikTok TechJam 2025 | Protecting voices, enabling creativity</p>
        <p>Built with cutting-edge AI: Real-time processing • Privacy-preserving • SOTA models</p>
      </footer>
    </div>
  );
};

export default VoiceShieldDashboard;
