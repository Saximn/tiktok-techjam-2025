"""
VoiceShield Project Entry Point
Quick test and demo launcher for the complete system
"""

import asyncio
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from examples.demo_tiktok_live import VoiceShieldDemo
from tests.test_voice_shield import *

def main():
    """Main entry point for VoiceShield"""
    # Handle Windows console encoding
    if os.name == 'nt':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    
    print("VoiceShield - Real-Time AI Voice Privacy Protection")
    print("=" * 60)
    print("Built for TikTok TechJam 2025")
    print()
    
    while True:
        print("Choose an option:")
        print("1. Run TikTok Live Demo")
        print("2. Run Test Suite") 
        print("3. Launch Mobile App (React Native)")
        print("4. Start Web Interface")
        print("5. Quick Performance Test")
        print("6. Exit")
        print()
        
        choice = input("Enter your choice (1-6): ").strip()
        print()
        
        if choice == "1":
            print("Starting TikTok Live Demo...")
            demo = VoiceShieldDemo()
            asyncio.run(demo.run_complete_demo())
            
        elif choice == "2":
            print("Running Test Suite...")
            run_tests()
            
        elif choice == "3":
            print("Mobile app requires Expo CLI to be installed")
            print("Run: npm install -g @expo/cli")
            print("Then: expo start")
            
        elif choice == "4":
            print("Web interface requires npm dependencies")
            print("Run: npm install && npm run dev")
            
        elif choice == "5":
            print("Running Quick Performance Test...")
            asyncio.run(quick_performance_test())
            
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-6.")
        
        print("\n" + "="*60 + "\n")

async def quick_performance_test():
    """Run a quick performance test of core VoiceShield functionality"""
    print("Initializing VoiceShield for performance test...")
    
    # Import required modules
    from core.voice_shield import VoiceShield, PrivacyMode, AudioChunk
    import numpy as np
    import time
    
    # Create VoiceShield instance
    voice_shield = VoiceShield(
        sample_rate=48000,
        chunk_size_ms=20,
        privacy_mode=PrivacyMode.PERSONAL
    )
    
    print("Loading AI models...")
    await voice_shield.initialize_models()
    
    print("Running performance test with 100 audio chunks...")
    
    # Performance test
    processing_times = []
    
    for i in range(100):
        # Create test audio chunk (20ms at 48kHz)
        audio_data = np.random.randn(960).astype(np.float32) * 0.1
        test_chunk = AudioChunk(
            data=audio_data,
            sample_rate=48000,
            timestamp=time.time(),
            duration_ms=20
        )
        
        # Time the processing
        start_time = time.perf_counter()
        processed_chunk, metrics = voice_shield.process_realtime_audio(test_chunk)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        processing_times.append(processing_time)
        
        if i % 20 == 0:
            print(f"  Processed {i+1}/100 chunks...")
    
    # Calculate statistics
    avg_latency = np.mean(processing_times)
    max_latency = np.max(processing_times)
    min_latency = np.min(processing_times)
    target_met = sum(1 for t in processing_times if t <= 50) / len(processing_times) * 100
    
    print("\nPerformance Test Results:")
    print(f"   Average latency: {avg_latency:.2f}ms")
    print(f"   Maximum latency: {max_latency:.2f}ms") 
    print(f"   Minimum latency: {min_latency:.2f}ms")
    print(f"   Target met (≤50ms): {target_met:.1f}%")
    
    if avg_latency <= 50:
        print("   Performance target achieved!")
    else:
        print("   Performance needs optimization")
    
    # Test different privacy modes
    print("\nTesting privacy modes...")
    for mode in [PrivacyMode.PERSONAL, PrivacyMode.MEETING, PrivacyMode.PUBLIC]:
        voice_shield.set_privacy_mode(mode)
        
        # Quick test
        test_chunk = AudioChunk(
            data=np.random.randn(960).astype(np.float32) * 0.2,
            sample_rate=48000,
            timestamp=time.time(),
            duration_ms=20
        )
        
        start_time = time.perf_counter()
        processed_chunk, metrics = voice_shield.process_realtime_audio(test_chunk)
        latency = (time.perf_counter() - start_time) * 1000
        
        print(f"   {mode.value.upper()}: {latency:.2f}ms, protection: {metrics.protection_level:.2f}")
    
    print("Performance test completed!")

def run_tests():
    """Run the test suite"""
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v"
        ], cwd=project_root, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
    except ImportError:
        print("pytest not found. Running basic tests...")
        # Run basic unittest instead
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover('tests', pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

if __name__ == "__main__":
    main()
