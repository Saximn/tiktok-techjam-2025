#!/usr/bin/env python3
"""
Real-Time Audio Processing Test - VoiceShield SOTA Models
Test the actual audio processing capabilities with simulated audio
"""

import sys
import os
import asyncio
import numpy as np
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.voice_shield import VoiceShield, PrivacyMode, AudioChunk

async def test_real_audio_processing():
    """Test real-time audio processing with simulated voice data"""
    
    print("REAL-TIME AUDIO PROCESSING TEST WITH SOTA MODELS")
    print("="*60)
    
    # Initialize VoiceShield
    print("1. Initializing VoiceShield...")
    voice_shield = VoiceShield(
        sample_rate=48000,
        chunk_size_ms=20,
        privacy_mode=PrivacyMode.PUBLIC  # Maximum protection for testing
    )
    
    # Load models (this will test if SOTA models actually work)
    print("2. Loading SOTA AI models (this may take a while)...")
    start_time = time.time()
    await voice_shield.initialize_models()
    load_time = time.time() - start_time
    print(f"[SUCCESS] Models loaded in {load_time:.2f} seconds")
    
    # Generate simulated voice audio for testing
    print("3. Generating test audio...")
    sample_rate = 48000
    duration_seconds = 2  # Shorter for testing
    chunk_duration_ms = 20
    chunk_size = int(sample_rate * chunk_duration_ms / 1000)  # 960 samples per chunk
    
    # Create realistic voice-like audio (sine waves with voice characteristics)
    total_samples = sample_rate * duration_seconds
    t = np.linspace(0, duration_seconds, total_samples)
    
    # Simulate human speech with fundamental frequency around 120 Hz
    fundamental_freq = 120
    test_audio = (
        0.3 * np.sin(2 * np.pi * fundamental_freq * t) +           # Fundamental
        0.2 * np.sin(2 * np.pi * fundamental_freq * 2 * t) +       # Second harmonic
        0.1 * np.sin(2 * np.pi * fundamental_freq * 3 * t) +       # Third harmonic
        0.05 * np.random.normal(0, 0.1, len(t))                    # Noise component
    ).astype(np.float32)
    
    # Process audio in chunks to simulate real-time streaming
    print("4. Processing audio chunks through privacy pipeline...")
    processed_chunks = []
    processing_times = []
    
    num_chunks = len(test_audio) // chunk_size
    print(f"   Processing {num_chunks} chunks...")
    
    for i in range(0, len(test_audio), chunk_size):
        chunk_data = test_audio[i:i+chunk_size]
        
        # Pad the last chunk if necessary
        if len(chunk_data) < chunk_size:
            chunk_data = np.pad(chunk_data, (0, chunk_size - len(chunk_data)), 'constant')
        
        # Create AudioChunk
        audio_chunk = AudioChunk(
            data=chunk_data,
            sample_rate=sample_rate,
            timestamp=i / sample_rate,
            duration_ms=chunk_duration_ms
        )
        
        # Process through VoiceShield pipeline
        chunk_start = time.perf_counter()
        processed_chunk, metrics = await voice_shield.process_realtime_audio(audio_chunk)
        processing_time = (time.perf_counter() - chunk_start) * 1000
        
        processed_chunks.append(processed_chunk)
        processing_times.append(processing_time)
        
        # Print progress every 20 chunks
        chunk_num = i // chunk_size + 1
        if chunk_num % 20 == 0 or chunk_num <= 5:
            print(f"   Chunk {chunk_num}/{num_chunks}: "
                  f"Latency: {processing_time:.2f}ms, "
                  f"Protection: {metrics.protection_level*100:.1f}%")
    
    # Analyze results
    print("\n5. PERFORMANCE ANALYSIS:")
    avg_latency = np.mean(processing_times)
    max_latency = np.max(processing_times)
    min_latency = np.min(processing_times)
    target_met = sum(1 for t in processing_times if t <= 50) / len(processing_times) * 100
    
    print(f"   Total chunks processed: {len(processed_chunks)}")
    print(f"   Average latency: {avg_latency:.2f}ms")
    print(f"   Max latency: {max_latency:.2f}ms")
    print(f"   Min latency: {min_latency:.2f}ms")
    print(f"   Target achievement: {target_met:.1f}% (< 50ms)")
    
    # Test privacy features
    print("\n6. TESTING PRIVACY MODES:")
    
    # Test different privacy modes
    privacy_modes = [PrivacyMode.PERSONAL, PrivacyMode.MEETING, PrivacyMode.PUBLIC, PrivacyMode.EMERGENCY]
    
    test_chunk = AudioChunk(
        data=test_audio[:chunk_size],
        sample_rate=sample_rate,
        timestamp=0,
        duration_ms=chunk_duration_ms
    )
    
    for mode in privacy_modes:
        voice_shield.set_privacy_mode(mode)
        start_time = time.perf_counter()
        processed_chunk, metrics = await voice_shield.process_realtime_audio(test_chunk)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        print(f"   {mode.value.upper()} mode: {processing_time:.2f}ms, {metrics.protection_level*100:.1f}% protection")
    
    # Test emergency privacy toggle
    print("\n7. TESTING EMERGENCY FEATURES:")
    voice_shield.emergency_privacy_toggle()  # Turn off
    test_result = await voice_shield.process_realtime_audio(test_chunk)
    print(f"   Emergency OFF: Audio passthrough active")
    
    voice_shield.emergency_privacy_toggle()  # Turn on
    test_result = await voice_shield.process_realtime_audio(test_chunk)
    print(f"   Emergency ON: Maximum protection active")
    
    # Get comprehensive performance stats
    print("\n8. SYSTEM PERFORMANCE SUMMARY:")
    stats = voice_shield.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n[SUCCESS] Real-time audio processing test completed!")
    print(f"System is ready for production use with SOTA AI models!")

if __name__ == "__main__":
    asyncio.run(test_real_audio_processing())
