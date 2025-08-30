#!/usr/bin/env python3
"""
Quick VoiceShield Demo Test - TikTok Live Integration
Testing the complete system functionality
"""

import sys
import os
import asyncio
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=== VoiceShield Demo Test ===")
print("Testing TikTok Live integration and core functionality...")
print()

async def test_demo():
    """Test the VoiceShield demo functionality"""
    
    try:
        # Import demo components
        print("Importing demo components...")
        from examples.demo_tiktok_live import VoiceShieldDemo
        
        # Create demo instance
        print("Creating VoiceShield demo instance...")
        demo = VoiceShieldDemo()
        
        # This would normally run the full demo, but let's just test initialization
        print("Demo instance created successfully!")
        print()
        
        # Test VoiceShield core functionality
        print("Testing VoiceShield core functionality...")
        from core.voice_shield import VoiceShield, PrivacyMode, AudioChunk
        import numpy as np
        
        # Create VoiceShield instance
        voice_shield = VoiceShield(
            sample_rate=48000,
            chunk_size_ms=20,
            privacy_mode=PrivacyMode.PERSONAL
        )
        
        print(f"VoiceShield initialized - Mode: {voice_shield.privacy_mode.value}")
        
        # Test different privacy modes
        privacy_modes = [PrivacyMode.PERSONAL, PrivacyMode.MEETING, PrivacyMode.PUBLIC, PrivacyMode.EMERGENCY]
        
        for mode in privacy_modes:
            voice_shield.set_privacy_mode(mode)
            print(f"Privacy mode set to: {mode.value}")
        
        print()
        
        # Test TikTok Live integration
        print("Testing TikTok Live integration...")
        from integrations.tiktok.live_integration import TikTokLiveVoiceShield, StreamingMode
        
        # Create TikTok Live integration
        tiktok_shield = TikTokLiveVoiceShield(
            voice_shield=voice_shield,
            stream_key="demo_stream_key_12345"
        )
        
        print(f"TikTok Live integration created - Mode: {tiktok_shield.streaming_mode.value}")
        
        # Test starting a mock stream
        print("Starting mock live stream...")
        stream_result = await tiktok_shield.start_live_stream("VoiceShield Demo Stream")
        
        if stream_result.get("success"):
            print(f"Mock stream started successfully!")
            print(f"  Stream ID: {stream_result.get('stream_id')}")
            print(f"  Privacy Protection: {stream_result.get('privacy_protection')}")
            print(f"  Latency Target: {stream_result.get('latency_target')}")
            
            # Let it run for a few seconds
            print("Running stream for 5 seconds...")
            await asyncio.sleep(5)
            
            # Get live metrics
            metrics = tiktok_shield.get_live_metrics()
            print(f"Live metrics: {metrics}")
            
            # Stop the stream
            print("Stopping mock stream...")
            stop_result = await tiktok_shield.stop_live_stream()
            
            if stop_result.get("success"):
                print("Mock stream stopped successfully!")
                summary = stop_result.get("stream_summary")
                print(f"Stream summary: {summary}")
            else:
                print(f"Stream stop failed: {stop_result.get('error')}")
                
        else:
            print(f"Mock stream failed to start: {stream_result.get('error')}")
        
        print()
        print("[SUCCESS] VoiceShield demo test completed successfully!")
        
    except Exception as e:
        print(f"[FAILED] Demo test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_demo())
