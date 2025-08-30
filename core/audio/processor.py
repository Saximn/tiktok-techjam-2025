"""
Audio Processing Utilities for VoiceShield
Real-time audio capture, processing, and streaming
"""

import numpy as np
import asyncio
from typing import Optional, Callable, Generator
import time
import threading
import queue
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 48000
    channels: int = 1
    chunk_duration_ms: int = 20
    buffer_size: int = 4096
    format_bit_depth: int = 16

class AudioBuffer:
    """Thread-safe circular audio buffer for real-time processing"""
    
    def __init__(self, max_size_seconds: float = 5.0, sample_rate: int = 48000):
        self.max_size = int(max_size_seconds * sample_rate)
        self.buffer = np.zeros(self.max_size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self.lock = threading.Lock()
        
    def write(self, data: np.ndarray):
        """Write audio data to buffer"""
        with self.lock:
            data_len = len(data)
            
            # Handle buffer wraparound
            if self.write_pos + data_len <= self.max_size:
                self.buffer[self.write_pos:self.write_pos + data_len] = data
            else:
                # Split write across buffer boundary
                first_part = self.max_size - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:data_len - first_part] = data[first_part:]
            
            self.write_pos = (self.write_pos + data_len) % self.max_size
            self.size = min(self.size + data_len, self.max_size)
    
    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read audio data from buffer"""
        with self.lock:
            if self.size < num_samples:
                return None
            
            # Handle buffer wraparound
            if self.read_pos + num_samples <= self.max_size:
                data = self.buffer[self.read_pos:self.read_pos + num_samples].copy()
            else:
                # Split read across buffer boundary
                first_part = self.max_size - self.read_pos
                data = np.concatenate([
                    self.buffer[self.read_pos:],
                    self.buffer[:num_samples - first_part]
                ])
            
            self.read_pos = (self.read_pos + num_samples) % self.max_size
            self.size -= num_samples
            return data

class RealTimeAudioProcessor:
    """Real-time audio processing pipeline"""
    
    def __init__(self, config: AudioConfig, processing_callback: Callable):
        self.config = config
        self.processing_callback = processing_callback
        self.audio_buffer = AudioBuffer(sample_rate=config.sample_rate)
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.input_thread = None
        
        # Performance tracking
        self.chunks_processed = 0
        self.total_processing_time = 0
        self.dropped_chunks = 0
        
        # Calculate chunk size
        self.chunk_size = int(config.sample_rate * config.chunk_duration_ms / 1000)
        
        logger.info(f"Audio processor initialized - Chunk size: {self.chunk_size} samples")

    def start_processing(self):
        """Start real-time audio processing"""
        if self.is_running:
            logger.warning("Audio processing already running")
            return
        
        self.is_running = True
        
        # Start async processing task
        import asyncio
        self.processing_task = asyncio.create_task(self._processing_loop())
        
        # Start input simulation thread (in real app, this would be mic input)
        self.input_thread = threading.Thread(
            target=self._input_simulation_loop,
            daemon=True, 
            name="AudioInput"
        )
        self.input_thread.start()
        
        logger.info("Real-time audio processing started")
    
    def stop_processing(self):
        """Stop audio processing"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        if self.input_thread:
            self.input_thread.join(timeout=2.0)
            
        logger.info("Audio processing stopped")
    
    async def _processing_loop(self):
        """Main audio processing loop"""
        logger.info("Processing loop started")
        
        while self.is_running:
            try:
                # Read audio chunk from buffer
                audio_data = self.audio_buffer.read(self.chunk_size)
                
                if audio_data is None:
                    # No data available, short sleep to prevent busy waiting
                    await asyncio.sleep(0.001)  # 1ms
                    continue
                
                # Process audio chunk
                start_time = time.perf_counter()
                
                # Create audio chunk object
                from ..voice_shield import AudioChunk
                chunk = AudioChunk(
                    data=audio_data,
                    sample_rate=self.config.sample_rate,
                    timestamp=time.time(),
                    duration_ms=self.config.chunk_duration_ms
                )
                
                # Call processing callback (VoiceShield) - await since it's async
                processed_chunk, metrics = await self.processing_callback(chunk)
                
                # Update performance stats
                processing_time = (time.perf_counter() - start_time) * 1000
                self.total_processing_time += processing_time
                self.chunks_processed += 1
                
                # Log performance issues
                if processing_time > self.config.chunk_duration_ms:
                    self.dropped_chunks += 1
                    logger.warning(f"Processing too slow: {processing_time:.2f}ms > {self.config.chunk_duration_ms}ms")
                
                # In real app, output processed audio to speakers/stream
                self._output_audio(processed_chunk, metrics)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                await asyncio.sleep(0.01)  # Brief pause on error
    
    def _input_simulation_loop(self):
        """Simulate audio input (replace with real mic input)"""
        logger.info("Audio input simulation started")
        
        chunk_duration_s = self.config.chunk_duration_ms / 1000.0
        
        while self.is_running:
            try:
                # Generate mock audio data (replace with real mic capture)
                audio_data = self._generate_mock_audio(self.chunk_size)
                
                # Write to buffer
                self.audio_buffer.write(audio_data)
                
                # Sleep for chunk duration to simulate real-time
                time.sleep(chunk_duration_s)
                
            except Exception as e:
                logger.error(f"Input error: {e}")
                time.sleep(0.01)
    
    def _generate_mock_audio(self, num_samples: int) -> np.ndarray:
        """Generate mock audio for testing"""
        # Create realistic audio simulation
        t = np.linspace(0, num_samples / self.config.sample_rate, num_samples)
        
        # Mix of frequencies to simulate speech
        base_freq = 200 + np.random.random() * 300  # 200-500 Hz base
        audio = np.sin(2 * np.pi * base_freq * t) * 0.3
        
        # Add harmonic content
        audio += np.sin(2 * np.pi * base_freq * 2 * t) * 0.1
        audio += np.sin(2 * np.pi * base_freq * 3 * t) * 0.05
        
        # Add noise to simulate real audio
        noise = np.random.normal(0, 0.02, num_samples)
        audio = audio + noise
        
        # Occasionally simulate silence
        if np.random.random() < 0.3:
            audio = audio * 0.1  # Very quiet
        
        return audio.astype(np.float32)
    
    def _output_audio(self, chunk, metrics):
        """Handle processed audio output"""
        # In real implementation, this would send to:
        # - Speakers for monitoring
        # - Streaming service (TikTok Live)
        # - Recording file
        
        if self.chunks_processed % 100 == 0:  # Log every 100 chunks
            avg_latency = self.total_processing_time / self.chunks_processed
            logger.info(f"Processed {self.chunks_processed} chunks, "
                       f"avg latency: {avg_latency:.2f}ms, "
                       f"protection: {metrics.protection_level:.2f}")
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        if self.chunks_processed == 0:
            return {"status": "No processing yet"}
        
        avg_processing_time = self.total_processing_time / self.chunks_processed
        drop_rate = (self.dropped_chunks / self.chunks_processed) * 100
        
        return {
            "chunks_processed": self.chunks_processed,
            "avg_processing_ms": round(avg_processing_time, 2),
            "dropped_chunks": self.dropped_chunks,
            "drop_rate_pct": round(drop_rate, 2),
            "is_running": self.is_running,
            "buffer_utilization": self.audio_buffer.size / self.audio_buffer.max_size
        }


# Audio utility functions

def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level"""
    if len(audio) == 0:
        return audio
    
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    # Convert target dB to linear scale
    target_rms = 10 ** (target_db / 20.0)
    
    # Apply normalization
    return audio * (target_rms / rms)

def detect_clipping(audio: np.ndarray, threshold: float = 0.95) -> bool:
    """Detect audio clipping"""
    return np.any(np.abs(audio) > threshold)

def apply_fade(audio: np.ndarray, fade_samples: int = 100) -> np.ndarray:
    """Apply fade in/out to prevent audio pops"""
    if len(audio) < fade_samples * 2:
        return audio
    
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    
    return audio


__all__ = ['AudioConfig', 'AudioBuffer', 'RealTimeAudioProcessor', 'normalize_audio', 'detect_clipping', 'apply_fade']
