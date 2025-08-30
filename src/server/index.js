/**
 * VoiceShield Express Server with WebSocket Support
 * Handles web interface serving and real-time communication with Python backend
 */

import express from 'express';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import WebSocket from 'ws';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const config = {
    port: process.env.PORT || 3001,  // Fixed: Use 3001 for server to avoid conflict with Vite dev server
    pythonBackendPort: 8001,
    isDevelopment: process.env.NODE_ENV !== 'production',
    staticPath: path.join(__dirname, '../../public'),
    buildPath: path.join(__dirname, '../../dist')
};

class VoiceShieldServer {
    constructor() {
        this.app = express();
        this.server = createServer(this.app);
        this.io = new SocketIOServer(this.server, {
            cors: {
                origin: config.isDevelopment ? "*" : "http://localhost:3000",
                methods: ["GET", "POST"]
            }
        });
        
        // Python backend connection
        this.pythonProcess = null;
        this.pythonWebSocket = null;
        
        // Connected clients
        this.clients = new Map();
        
        // Real-time data storage
        this.currentStreamData = {
            isStreaming: false,
            viewerCount: 0,
            streamDuration: 0,
            privacyAlerts: 0,
            piiBlocked: 0,
            backgroundVoicesFiltered: 0,
            privacyMode: 'personal',
            protectionLevel: 0.6
        };
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
        this.startPythonBackend();
    }

    setupMiddleware() {
        // Security middleware
        this.app.use(helmet({
            contentSecurityPolicy: {
                directives: {
                    defaultSrc: ["'self'"],
                    scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'", "https://fonts.googleapis.com"],
                    styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
                    fontSrc: ["'self'", "https://fonts.gstatic.com"],
                    imgSrc: ["'self'", "data:", "https:"],
                    connectSrc: ["'self'", "ws:", "wss:"]
                }
            }
        }));

        // CORS middleware
        this.app.use(cors({
            origin: config.isDevelopment ? "*" : "http://localhost:3000",
            credentials: true
        }));

        // JSON parsing
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true }));

        // Static file serving
        if (config.isDevelopment) {
            this.app.use(express.static(config.staticPath));
        } else {
            this.app.use(express.static(config.buildPath));
        }
    }

    setupRoutes() {
        // Health check endpoint
        this.app.get('/api/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                uptime: process.uptime(),
                pythonBackendStatus: this.pythonProcess ? 'running' : 'stopped',
                connectedClients: this.clients.size
            });
        });

        // VoiceShield status endpoint
        this.app.get('/api/voiceshield/status', (req, res) => {
            res.json({
                ...this.currentStreamData,
                timestamp: new Date().toISOString()
            });
        });

        // Privacy mode endpoint
        this.app.post('/api/voiceshield/privacy-mode', (req, res) => {
            const { mode } = req.body;
            if (!['personal', 'meeting', 'public', 'emergency'].includes(mode)) {
                return res.status(400).json({ error: 'Invalid privacy mode' });
            }

            this.currentStreamData.privacyMode = mode;
            this.broadcastToClients('privacyModeChanged', { mode });
            
            // Send to Python backend
            this.sendToPythonBackend({
                type: 'change_privacy_mode',
                mode: mode
            });

            res.json({ success: true, mode });
        });

        // Protection level endpoint
        this.app.post('/api/voiceshield/protection-level', (req, res) => {
            const { level } = req.body;
            if (typeof level !== 'number' || level < 0 || level > 1) {
                return res.status(400).json({ error: 'Invalid protection level' });
            }

            this.currentStreamData.protectionLevel = level;
            this.broadcastToClients('protectionLevelChanged', { level });
            
            // Send to Python backend
            this.sendToPythonBackend({
                type: 'adjust_protection_level',
                level: level
            });

            res.json({ success: true, level });
        });

        // TikTok Live streaming endpoints
        this.app.post('/api/tiktok/start-stream', async (req, res) => {
            try {
                const { title = 'Live Stream' } = req.body;
                
                // Send to Python backend
                const response = await this.sendToPythonBackendAsync({
                    type: 'start_live_stream',
                    title: title
                });

                if (response.success) {
                    this.currentStreamData.isStreaming = true;
                    this.broadcastToClients('streamStarted', { title, streamId: response.streamId });
                    res.json(response);
                } else {
                    res.status(500).json({ error: response.error });
                }
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/tiktok/stop-stream', async (req, res) => {
            try {
                const response = await this.sendToPythonBackendAsync({
                    type: 'stop_live_stream'
                });

                this.currentStreamData.isStreaming = false;
                this.broadcastToClients('streamStopped', response);
                res.json(response);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Emergency privacy stop
        this.app.post('/api/voiceshield/emergency-stop', (req, res) => {
            this.sendToPythonBackend({
                type: 'emergency_privacy_stop'
            });

            this.broadcastToClients('emergencyStop', {
                message: 'Emergency privacy stop activated',
                timestamp: new Date().toISOString()
            });

            res.json({ success: true, message: 'Emergency stop activated' });
        });

        // Serve React app for all other routes
        this.app.get('*', (req, res) => {
            const indexPath = config.isDevelopment 
                ? path.join(config.staticPath, 'index.html')
                : path.join(config.buildPath, 'index.html');
            res.sendFile(indexPath);
        });
    }

    setupWebSocket() {
        this.io.on('connection', (socket) => {
            console.log(`Client connected: ${socket.id}`);
            
            // Store client
            this.clients.set(socket.id, socket);

            // Send initial state
            socket.emit('voiceShieldStatus', {
                mode: this.currentStreamData.privacyMode,
                protectionLevel: this.currentStreamData.protectionLevel,
                isStreaming: this.currentStreamData.isStreaming
            });

            socket.emit('streamMetrics', this.currentStreamData);

            // Handle client events
            socket.on('changePrivacyMode', (mode) => {
                this.handlePrivacyModeChange(mode);
            });

            socket.on('adjustProtectionLevel', (level) => {
                this.handleProtectionLevelChange(level);
            });

            socket.on('startLiveStream', async (data) => {
                await this.handleStartLiveStream(data, socket);
            });

            socket.on('stopLiveStream', async () => {
                await this.handleStopLiveStream(socket);
            });

            socket.on('emergencyPrivacyStop', () => {
                this.handleEmergencyStop();
            });

            // Handle disconnect
            socket.on('disconnect', () => {
                console.log(`Client disconnected: ${socket.id}`);
                this.clients.delete(socket.id);
            });
        });
    }

    async startPythonBackend() {
        try {
            console.log('Starting Python VoiceShield backend...');
            
            // Start Python process
            this.pythonProcess = spawn('python', [
                path.join(__dirname, '../../python_server.py'),
                '--port', config.pythonBackendPort.toString()
            ], {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: path.join(__dirname, '../..')
            });

            this.pythonProcess.stdout.on('data', (data) => {
                console.log(`Python Backend: ${data.toString()}`);
            });

            this.pythonProcess.stderr.on('data', (data) => {
                console.error(`Python Backend Error: ${data.toString()}`);
            });

            this.pythonProcess.on('close', (code) => {
                console.log(`Python backend exited with code ${code}`);
                this.pythonProcess = null;
            });

            // Wait a moment for Python server to start
            await new Promise(resolve => setTimeout(resolve, 3000));

            // Connect to Python WebSocket
            this.connectToPythonWebSocket();

        } catch (error) {
            console.error('Failed to start Python backend:', error);
        }
    }

    connectToPythonWebSocket() {
        try {
            this.pythonWebSocket = new WebSocket(`ws://localhost:${config.pythonBackendPort}/ws`);

            this.pythonWebSocket.on('open', () => {
                console.log('Connected to Python backend WebSocket');
            });

            this.pythonWebSocket.on('message', (data) => {
                const message = JSON.parse(data.toString());
                this.handlePythonMessage(message);
            });

            this.pythonWebSocket.on('error', (error) => {
                console.error('Python WebSocket error:', error);
            });

            this.pythonWebSocket.on('close', () => {
                console.log('Python WebSocket connection closed');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    this.connectToPythonWebSocket();
                }, 5000);
            });

        } catch (error) {
            console.error('Failed to connect to Python WebSocket:', error);
        }
    }

    handlePythonMessage(message) {
        switch (message.type) {
            case 'audio_data':
                this.broadcastToClients('audioData', message.data);
                break;
                
            case 'spectrogram_update':
                this.broadcastToClients('spectrogramUpdate', message.data);
                break;
                
            case 'speaker_update':
                this.broadcastToClients('speakerUpdate', message.speakers);
                break;
                
            case 'privacy_alert':
                this.currentStreamData.privacyAlerts++;
                this.broadcastToClients('privacyAlert', {
                    type: message.alertType,
                    message: message.message,
                    severity: message.severity || 'warning'
                });
                break;
                
            case 'pii_blocked':
                this.currentStreamData.piiBlocked++;
                this.broadcastToClients('piiBlocked', message);
                break;
                
            case 'background_voice_filtered':
                this.currentStreamData.backgroundVoicesFiltered++;
                this.broadcastToClients('backgroundVoiceFiltered', message);
                break;
                
            case 'stream_metrics':
                Object.assign(this.currentStreamData, message.data);
                this.broadcastToClients('streamMetrics', this.currentStreamData);
                break;
        }
    }

    // WebSocket event handlers
    handlePrivacyModeChange(mode) {
        this.currentStreamData.privacyMode = mode;
        this.broadcastToClients('voiceShieldStatus', {
            mode: mode,
            protectionLevel: this.currentStreamData.protectionLevel
        });
        
        this.sendToPythonBackend({
            type: 'change_privacy_mode',
            mode: mode
        });
    }

    handleProtectionLevelChange(level) {
        this.currentStreamData.protectionLevel = level;
        this.broadcastToClients('voiceShieldStatus', {
            mode: this.currentStreamData.privacyMode,
            protectionLevel: level
        });
        
        this.sendToPythonBackend({
            type: 'adjust_protection_level',
            level: level
        });
    }

    async handleStartLiveStream(data, socket) {
        try {
            const response = await this.sendToPythonBackendAsync({
                type: 'start_live_stream',
                title: data.title || 'Live Stream'
            });

            if (response.success) {
                this.currentStreamData.isStreaming = true;
                this.broadcastToClients('streamStarted', response);
                socket.emit('streamResponse', response);
            } else {
                socket.emit('streamResponse', { error: response.error });
            }
        } catch (error) {
            socket.emit('streamResponse', { error: error.message });
        }
    }

    async handleStopLiveStream(socket) {
        try {
            const response = await this.sendToPythonBackendAsync({
                type: 'stop_live_stream'
            });

            this.currentStreamData.isStreaming = false;
            this.broadcastToClients('streamStopped', response);
            socket.emit('streamResponse', response);
        } catch (error) {
            socket.emit('streamResponse', { error: error.message });
        }
    }

    handleEmergencyStop() {
        this.sendToPythonBackend({
            type: 'emergency_privacy_stop'
        });

        this.broadcastToClients('emergencyStop', {
            message: 'Emergency privacy stop activated',
            timestamp: new Date().toISOString()
        });
    }

    // Utility methods
    broadcastToClients(event, data) {
        this.clients.forEach(client => {
            client.emit(event, data);
        });
    }

    sendToPythonBackend(message) {
        if (this.pythonWebSocket && this.pythonWebSocket.readyState === WebSocket.OPEN) {
            this.pythonWebSocket.send(JSON.stringify(message));
        } else {
            console.warn('Python WebSocket not connected, message not sent:', message);
        }
    }

    sendToPythonBackendAsync(message) {
        return new Promise((resolve, reject) => {
            const messageId = Date.now().toString();
            const messageWithId = { ...message, id: messageId };

            // Set up timeout
            const timeout = setTimeout(() => {
                reject(new Error('Python backend response timeout'));
            }, 10000);

            // Set up response handler
            const responseHandler = (data) => {
                const response = JSON.parse(data.toString());
                if (response.id === messageId) {
                    clearTimeout(timeout);
                    this.pythonWebSocket.off('message', responseHandler);
                    resolve(response);
                }
            };

            if (this.pythonWebSocket && this.pythonWebSocket.readyState === WebSocket.OPEN) {
                this.pythonWebSocket.on('message', responseHandler);
                this.pythonWebSocket.send(JSON.stringify(messageWithId));
            } else {
                clearTimeout(timeout);
                reject(new Error('Python WebSocket not connected'));
            }
        });
    }

    start() {
        this.server.listen(config.port, () => {
            console.log(`🛡️  VoiceShield Server running on http://localhost:${config.port}`);
            console.log(`📱 Environment: ${config.isDevelopment ? 'Development' : 'Production'}`);
            console.log(`🔗 Python Backend Port: ${config.pythonBackendPort}`);
            console.log(`📊 Real-time WebSocket ready for TikTok Live integration`);
        });

        // Graceful shutdown
        process.on('SIGTERM', () => {
            console.log('SIGTERM received, shutting down gracefully...');
            this.server.close(() => {
                if (this.pythonProcess) {
                    this.pythonProcess.kill();
                }
                process.exit(0);
            });
        });
    }
}

// Start the server
const server = new VoiceShieldServer();
server.start();

export default VoiceShieldServer;
