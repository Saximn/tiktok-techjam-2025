#!/usr/bin/env node
/**
 * Quick Development Server for VoiceShield
 * Starts both the web interface and mock backend
 */

import express from 'express';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = createServer(app);
const io = new SocketIOServer(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

const PORT = 3000;

// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Mock data for demonstration
let streamData = {
    isStreaming: false,
    viewerCount: 0,
    streamDuration: 0,
    privacyAlerts: 0,
    piiBlocked: 0,
    backgroundVoicesFiltered: 0,
    privacyMode: 'personal',
    protectionLevel: 0.6
};

// API Routes
app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        connectedClients: io.engine.clientsCount
    });
});

app.get('/api/voiceshield/status', (req, res) => {
    res.json({
        ...streamData,
        timestamp: new Date().toISOString()
    });
});

app.post('/api/voiceshield/privacy-mode', (req, res) => {
    const { mode } = req.body;
    if (['personal', 'meeting', 'public', 'emergency'].includes(mode)) {
        streamData.privacyMode = mode;
        io.emit('privacyModeChanged', { mode });
        res.json({ success: true, mode });
    } else {
        res.status(400).json({ error: 'Invalid privacy mode' });
    }
});

app.post('/api/voiceshield/protection-level', (req, res) => {
    const { level } = req.body;
    if (typeof level === 'number' && level >= 0 && level <= 1) {
        streamData.protectionLevel = level;
        io.emit('protectionLevelChanged', { level });
        res.json({ success: true, level });
    } else {
        res.status(400).json({ error: 'Invalid protection level' });
    }
});

app.post('/api/tiktok/start-stream', (req, res) => {
    const { title = 'Live Stream' } = req.body;
    streamData.isStreaming = true;
    streamData.viewerCount = Math.floor(Math.random() * 100) + 1;
    
    io.emit('streamStarted', { 
        title, 
        streamId: `live_${Date.now()}`,
        success: true 
    });
    
    res.json({ 
        success: true, 
        streamId: `live_${Date.now()}`,
        title 
    });
});

app.post('/api/tiktok/stop-stream', (req, res) => {
    streamData.isStreaming = false;
    streamData.viewerCount = 0;
    
    const summary = {
        duration_seconds: streamData.streamDuration,
        max_viewers: streamData.viewerCount,
        privacy_alerts: streamData.privacyAlerts,
        pii_blocked: streamData.piiBlocked
    };
    
    io.emit('streamStopped', { success: true, stream_summary: summary });
    res.json({ success: true, stream_summary: summary });
});

app.post('/api/voiceshield/emergency-stop', (req, res) => {
    io.emit('emergencyStop', {
        message: 'Emergency privacy stop activated',
        timestamp: new Date().toISOString()
    });
    res.json({ success: true, message: 'Emergency stop activated' });
});

// WebSocket handling
io.on('connection', (socket) => {
    console.log(`Client connected: ${socket.id}`);

    // Send initial state
    socket.emit('voiceShieldStatus', {
        mode: streamData.privacyMode,
        protectionLevel: streamData.protectionLevel,
        isStreaming: streamData.isStreaming
    });

    socket.emit('streamMetrics', streamData);

    // Handle client events
    socket.on('changePrivacyMode', (mode) => {
        if (['personal', 'meeting', 'public', 'emergency'].includes(mode)) {
            streamData.privacyMode = mode;
            io.emit('voiceShieldStatus', {
                mode: mode,
                protectionLevel: streamData.protectionLevel
            });
        }
    });

    socket.on('adjustProtectionLevel', (level) => {
        if (typeof level === 'number' && level >= 0 && level <= 1) {
            streamData.protectionLevel = level;
            io.emit('voiceShieldStatus', {
                mode: streamData.privacyMode,
                protectionLevel: level
            });
        }
    });

    socket.on('startLiveStream', async (data) => {
        streamData.isStreaming = true;
        streamData.viewerCount = Math.floor(Math.random() * 100) + 1;
        
        const response = {
            success: true,
            streamId: `live_${Date.now()}`,
            title: data.title || 'Live Stream'
        };
        
        io.emit('streamStarted', response);
        socket.emit('streamResponse', response);
    });

    socket.on('stopLiveStream', async () => {
        streamData.isStreaming = false;
        const response = { success: true };
        io.emit('streamStopped', response);
        socket.emit('streamResponse', response);
    });

    socket.on('emergencyPrivacyStop', () => {
        io.emit('emergencyStop', {
            message: 'Emergency privacy stop activated',
            timestamp: new Date().toISOString()
        });
    });

    socket.on('disconnect', () => {
        console.log(`Client disconnected: ${socket.id}`);
    });
});

// Mock real-time data simulation
setInterval(() => {
    if (streamData.isStreaming) {
        // Simulate viewer changes
        streamData.viewerCount += Math.floor(Math.random() * 10) - 5;
        streamData.viewerCount = Math.max(0, streamData.viewerCount);
        
        // Simulate stream duration
        streamData.streamDuration += 1;
        
        // Generate mock audio data for visualization
        const mockAudioData = Array.from({ length: 512 }, () => 
            (Math.random() - 0.5) * 0.5 * (streamData.protectionLevel + 0.1)
        );
        
        // Generate mock spectrogram data
        const mockSpectrogram = Array.from({ length: 64 }, () => Math.random() * 0.8);
        
        // Generate mock speaker data
        const mockSpeakers = [
            {
                id: 'speaker_1',
                isMainSpeaker: true,
                confidence: 0.9 + Math.random() * 0.1,
                privacyLevel: streamData.protectionLevel
            }
        ];
        
        // Broadcast real-time data
        io.emit('audioData', mockAudioData);
        io.emit('spectrogramUpdate', mockSpectrogram);
        io.emit('speakerUpdate', mockSpeakers);
        io.emit('streamMetrics', streamData);
        
        // Occasionally trigger privacy events
        if (Math.random() < 0.05) { // 5% chance per second
            streamData.privacyAlerts++;
            io.emit('privacyAlert', {
                type: 'info',
                message: 'Background voice filtered automatically',
                severity: 'info'
            });
        }
        
        if (Math.random() < 0.02) { // 2% chance per second
            streamData.piiBlocked++;
            io.emit('privacyAlert', {
                type: 'warning',
                message: 'Personal information detected and masked',
                severity: 'warning'
            });
        }
    }
}, 1000);

// Serve React app for all routes
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
server.listen(PORT, () => {
    console.log(`🛡️  VoiceShield Web Interface running at http://localhost:${PORT}`);
    console.log(`🌐 Open your browser to see the live demo!`);
    console.log(`📊 Real-time WebSocket connections: ws://localhost:${PORT}`);
    console.log('');
    console.log('Features available:');
    console.log('  ✅ Real-time privacy shield visualization');
    console.log('  ✅ Dynamic privacy mode switching');
    console.log('  ✅ Live audio waveform and spectrogram');
    console.log('  ✅ TikTok Live streaming simulation');
    console.log('  ✅ Emergency privacy controls');
    console.log('  ✅ Real-time privacy metrics');
    console.log('');
    console.log('🎯 This is a full demonstration of VoiceShield capabilities!');
});

export default app;
