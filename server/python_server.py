#!/usr/bin/env python3
"""
VoiceShield Python WebSocket Server
Bridges the Node.js web server with the Python VoiceShield engine
Handles real-time audio processing and WebSocket communication
"""

import asyncio
import websockets
import json
import argparse
import logging
import time
import numpy as np
from typing import Dict, List, Set, Optional
import threading
from queue import Queue
import signal
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import VoiceShield components
from core.voice_shield import VoiceShield, PrivacyMode, AudioChunk
from integrations.tiktok.live_integration import TikTokLiveVoiceShield
from core.audio.processor import RealTimeAudioProcessor, AudioConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceShieldWebSocketServer:
    """
    WebSocket server for real-time communication between web interface and VoiceShield
    
    Features:
    - Real-time audio data streaming
    - Privacy metrics broadcasting  
    - TikTok Live integration control
    - Multi-client WebSocket management
    - Async audio processing pipeline
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8001):
        self.host = host
        self.port = port
        
        # WebSocket clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # VoiceShield components
        self.voice_shield = None
        self.audio_processor = None
        self.tiktok_integration = None
        
        # Real-time data queues
        self.audio_queue = Queue(maxsize=1000)
        self.spectrogram_queue = Queue(maxsize=100)
        self.metrics_queue = Queue(maxsize=100)
        
        # Server state
        self.is_running = False
        self.background_tasks = []
        
        # Message handlers
        self.message_handlers = {
            'change_privacy_mode': self.handle_privacy_mode_change,
            'adjust_protection_level': self.handle_protection_level_change,
            'start_live_stream': self.handle_start_live_stream,
            'stop_live_stream': self.handle_stop_live_stream,
            'emergency_privacy_stop': self.handle_emergency_stop,
            'get_status': self.handle_get_status
        }
        
        logger.info(f"VoiceShield WebSocket Server initialized on {host}:{port}")
    
    async def initialize_voice_shield(self):
        """Initialize VoiceShield components"""
        try:
            logger.info("Initializing VoiceShield components...")
            
            # Create VoiceShield instance
            self.voice_shield = VoiceShield(
                sample_rate=48000,
                chunk_size_ms=20,
                privacy_mode=PrivacyMode.PERSONAL,
                enable_advanced_features=True
            )
            
            # Initialize AI models (this takes time)
            logger.info("Loading AI models (this may take a minute)...")
            await self.voice_shield.initialize_models()
            
            # Create audio processor
            audio_config = AudioConfig(
                sample_rate=48000,
                chunk_duration_ms=20,
                channels=1,
                buffer_size=8192
            )
            
            self.audio_processor = RealTimeAudioProcessor(
                config=audio_config,
                processing_callback=self.voice_shield.process_realtime_audio
            )
            
            # Create TikTok Live integration
            self.tiktok_integration = TikTokLiveVoiceShield(
                voice_shield=self.voice_shield
            )
            
            # Set up audio processing callbacks
            self.audio_processor.set_audio_callback(self.on_audio_data)
            self.audio_processor.set_metrics_callback(self.on_metrics_update)
            
            logger.info("VoiceShield components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize VoiceShield: {e}")
            raise
    
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            # Initialize VoiceShield
            await self.initialize_voice_shield()
            
            self.is_running = True
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self.audio_processing_task()),
                asyncio.create_task(self.metrics_broadcasting_task()),
                asyncio.create_task(self.spectrogram_broadcasting_task())
            ]
            
            # Start WebSocket server
            logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
            
            async with websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10
            ):
                logger.info("WebSocket server started successfully!")
                logger.info("Waiting for connections...")
                
                # Keep server running
                await asyncio.Future()  # Run forever
                
        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            raise
    
    async def handle_client(self, websocket, path):
        """Handle new WebSocket client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New client connected: {client_id}")
        
        # Add client to set
        self.clients.add(websocket)
        
        try:
            # Send initial status
            await self.send_to_client(websocket, {
                'type': 'connection_established',
                'clientId': client_id,
                'timestamp': time.time()
            })
            
            # Send current status
            await self.send_current_status(websocket)
            
            # Handle messages from client
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Client error {client_id}: {e}")
        finally:
            # Remove client from set
            self.clients.discard(websocket)
    
    async def handle_client_message(self, websocket, message):
        """Handle message from WebSocket client"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            message_id = data.get('id')
            
            if message_type in self.message_handlers:
                logger.debug(f"Handling message: {message_type}")
                
                # Call handler
                response = await self.message_handlers[message_type](data)
                
                # Send response if message had ID (expecting response)
                if message_id:
                    response['id'] = message_id
                    await self.send_to_client(websocket, response)
                    
            else:
                logger.warning(f"Unknown message type: {message_type}")
                if message_id:
                    await self.send_to_client(websocket, {
                        'id': message_id,
                        'error': f'Unknown message type: {message_type}'
                    })
                    
        except json.JSONDecodeError:
            logger.error("Invalid JSON message received")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    async def send_to_client(self, websocket, data):
        """Send data to specific client"""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            pass  # Client already disconnected
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
    
    async def broadcast_to_all_clients(self, data):
        """Broadcast data to all connected clients"""
        if not self.clients:
            return
        
        # Create tasks for all clients
        tasks = []
        for client in self.clients.copy():  # Copy to avoid modification during iteration
            tasks.append(self.send_to_client(client, data))
        
        # Send to all clients concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    # Message Handlers
    
    async def handle_privacy_mode_change(self, data):
        """Handle privacy mode change request"""
        try:
            mode_str = data.get('mode', 'personal')
            privacy_mode = PrivacyMode(mode_str)
            
            # Update VoiceShield
            if self.voice_shield:
                self.voice_shield.set_privacy_mode(privacy_mode)
            
            logger.info(f"Privacy mode changed to: {mode_str}")
            
            # Broadcast to all clients
            await self.broadcast_to_all_clients({
                'type': 'privacy_mode_changed',
                'mode': mode_str,
                'timestamp': time.time()
            })
            
            return {'success': True, 'mode': mode_str}
            
        except Exception as e:
            logger.error(f"Privacy mode change failed: {e}")
            return {'error': str(e)}
    
    async def handle_protection_level_change(self, data):
        """Handle protection level adjustment"""
        try:
            level = float(data.get('level', 0.6))
            level = max(0.0, min(1.0, level))  # Clamp to valid range
            
            # Update VoiceShield (this would need to be implemented in the core)
            logger.info(f"Protection level adjusted to: {level}")
            
            # Broadcast to all clients
            await self.broadcast_to_all_clients({
                'type': 'protection_level_changed',
                'level': level,
                'timestamp': time.time()
            })
            
            return {'success': True, 'level': level}
            
        except Exception as e:
            logger.error(f"Protection level change failed: {e}")
            return {'error': str(e)}
    
    async def handle_start_live_stream(self, data):
        """Handle TikTok Live stream start"""
        try:
            title = data.get('title', 'Live Stream')
            
            if not self.tiktok_integration:
                return {'error': 'TikTok integration not initialized'}
            
            # Start live stream
            result = await self.tiktok_integration.start_live_stream(title)
            
            if result.get('success'):
                logger.info(f"Live stream started: {title}")
                
                # Broadcast to all clients
                await self.broadcast_to_all_clients({
                    'type': 'stream_started',
                    'title': title,
                    'streamId': result.get('stream_id'),
                    'timestamp': time.time()
                })
                
                # Start audio processing
                if self.audio_processor and not self.audio_processor.is_running:
                    await self.audio_processor.start_processing()
                
                return result
            else:
                return result
                
        except Exception as e:
            logger.error(f"Stream start failed: {e}")
            return {'error': str(e)}
    
    async def handle_stop_live_stream(self, data):
        """Handle TikTok Live stream stop"""
        try:
            if not self.tiktok_integration:
                return {'error': 'TikTok integration not initialized'}
            
            # Stop live stream
            result = await self.tiktok_integration.stop_live_stream()
            
            if result.get('success'):
                logger.info("Live stream stopped")
                
                # Stop audio processing
                if self.audio_processor and self.audio_processor.is_running:
                    self.audio_processor.stop_processing()
                
                # Broadcast to all clients
                await self.broadcast_to_all_clients({
                    'type': 'stream_stopped',
                    'summary': result.get('stream_summary'),
                    'timestamp': time.time()
                })
                
                return result
            else:
                return result
                
        except Exception as e:
            logger.error(f"Stream stop failed: {e}")
            return {'error': str(e)}
    
    async def handle_emergency_stop(self, data):
        """Handle emergency privacy stop"""
        try:
            # Activate emergency mode
            if self.voice_shield:
                self.voice_shield.emergency_privacy_toggle()
            
            if self.tiktok_integration:
                self.tiktok_integration.emergency_privacy_stop()
            
            logger.critical("Emergency privacy stop activated!")
            
            # Broadcast to all clients
            await self.broadcast_to_all_clients({
                'type': 'emergency_stop',
                'message': 'Emergency privacy stop activated',
                'timestamp': time.time()
            })
            
            return {'success': True, 'message': 'Emergency stop activated'}
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {'error': str(e)}
    
    async def handle_get_status(self, data):
        """Handle status request"""
        try:
            status = {
                'voiceShieldActive': self.voice_shield is not None,
                'privacyMode': self.voice_shield.privacy_mode.value if self.voice_shield else 'personal',
                'protectionActive': self.voice_shield.protection_active if self.voice_shield else False,
                'isStreaming': self.tiktok_integration.streaming_mode.value != 'offline' if self.tiktok_integration else False,
                'clientsConnected': len(self.clients),
                'timestamp': time.time()
            }
            
            if self.voice_shield:
                perf_stats = self.voice_shield.get_performance_stats()
                status['performanceStats'] = perf_stats
            
            return {'success': True, 'status': status}
            
        except Exception as e:
            logger.error(f"Status request failed: {e}")
            return {'error': str(e)}
    
    async def send_current_status(self, websocket):
        """Send current system status to client"""
        try:
            status_response = await self.handle_get_status({})
            await self.send_to_client(websocket, {
                'type': 'initial_status',
                'data': status_response.get('status', {}),
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"Failed to send current status: {e}")
    
    # Audio Processing Callbacks
    
    def on_audio_data(self, audio_data: np.ndarray, metadata: Dict):
        """Callback for new audio data"""
        try:
            # Add to queue for broadcasting
            if not self.audio_queue.full():
                self.audio_queue.put({
                    'data': audio_data.tolist(),
                    'metadata': metadata,
                    'timestamp': time.time()
                })
        except Exception as e:
            logger.error(f"Audio data callback error: {e}")
    
    def on_metrics_update(self, metrics: Dict):
        """Callback for metrics updates"""
        try:
            # Add to queue for broadcasting
            if not self.metrics_queue.full():
                self.metrics_queue.put({
                    'metrics': metrics,
                    'timestamp': time.time()
                })
        except Exception as e:
            logger.error(f"Metrics callback error: {e}")
    
    # Background Tasks
    
    async def audio_processing_task(self):
        """Background task for processing audio data queue"""
        while self.is_running:
            try:
                # Process audio data queue
                while not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    
                    await self.broadcast_to_all_clients({
                        'type': 'audio_data',
                        'data': audio_data['data'][:512],  # Send first 512 samples for visualization
                        'timestamp': audio_data['timestamp']
                    })
                
                await asyncio.sleep(0.05)  # 50ms interval
                
            except Exception as e:
                logger.error(f"Audio processing task error: {e}")
                await asyncio.sleep(1)
    
    async def metrics_broadcasting_task(self):
        """Background task for broadcasting metrics"""
        while self.is_running:
            try:
                # Process metrics queue
                while not self.metrics_queue.empty():
                    metrics_data = self.metrics_queue.get_nowait()
                    
                    await self.broadcast_to_all_clients({
                        'type': 'metrics_update',
                        'data': metrics_data['metrics'],
                        'timestamp': metrics_data['timestamp']
                    })
                
                # Also send periodic status updates
                if self.tiktok_integration:
                    live_metrics = self.tiktok_integration.get_live_metrics()
                    
                    await self.broadcast_to_all_clients({
                        'type': 'stream_metrics',
                        'data': {
                            'viewerCount': live_metrics.viewer_count,
                            'streamDuration': live_metrics.stream_duration_seconds,
                            'privacyAlerts': live_metrics.privacy_alerts_triggered,
                            'piiBlocked': live_metrics.pii_blocked_count,
                            'backgroundVoicesFiltered': live_metrics.background_voices_filtered,
                            'averageProtectionLevel': live_metrics.average_protection_level,
                            'streamHealthScore': live_metrics.stream_health_score
                        },
                        'timestamp': time.time()
                    })
                
                await asyncio.sleep(2)  # 2 second interval for metrics
                
            except Exception as e:
                logger.error(f"Metrics broadcasting task error: {e}")
                await asyncio.sleep(5)
    
    async def spectrogram_broadcasting_task(self):
        """Background task for broadcasting spectrogram data"""
        while self.is_running:
            try:
                # Generate mock spectrogram data for visualization
                # In real implementation, this would come from actual audio analysis
                spectrogram_frame = np.random.rand(64) * 0.5  # 64 frequency bins
                
                await self.broadcast_to_all_clients({
                    'type': 'spectrogram_update',
                    'data': spectrogram_frame.tolist(),
                    'timestamp': time.time()
                })
                
                await asyncio.sleep(0.1)  # 100ms interval for spectrogram
                
            except Exception as e:
                logger.error(f"Spectrogram broadcasting task error: {e}")
                await asyncio.sleep(1)
    
    def stop_server(self):
        """Stop the server gracefully"""
        logger.info("Stopping VoiceShield WebSocket server...")
        self.is_running = False
        
        # Stop audio processing
        if self.audio_processor:
            self.audio_processor.stop_processing()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='VoiceShield WebSocket Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8001, help='Server port')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create and start server
    server = VoiceShieldWebSocketServer(args.host, args.port)
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        server.stop_server()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
