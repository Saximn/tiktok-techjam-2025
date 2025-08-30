#!/usr/bin/env python3
"""
Simple test script to check VoiceShield imports and basic functionality
"""

import sys
import os
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=== VoiceShield Import Test ===")
print(f"Python version: {sys.version}")
print(f"Project root: {project_root}")
print()

def test_import(module_name, description):
    """Test importing a module"""
    try:
        print(f"Testing {description}...")
        exec(f"import {module_name}")
        print(f"[SUCCESS] {description}")
        return True
    except Exception as e:
        print(f"[FAILED] {description} - Error: {e}")
        print(f"   Error details: {traceback.format_exc().split(chr(10))[-3:-1]}")
        return False

# Test basic Python libraries
print("--- Basic Libraries ---")
test_import("numpy", "NumPy")
test_import("torch", "PyTorch") 
test_import("asyncio", "AsyncIO")
print()

# Test VoiceShield core components
print("--- VoiceShield Core ---")
core_success = test_import("core.voice_shield", "Core VoiceShield Engine")

if core_success:
    print("Testing VoiceShield initialization...")
    try:
        from core.voice_shield import VoiceShield, PrivacyMode, AudioChunk
        
        # Create VoiceShield instance
        voice_shield = VoiceShield(
            sample_rate=48000,
            chunk_size_ms=20,
            privacy_mode=PrivacyMode.PERSONAL
        )
        
        print("[SUCCESS] VoiceShield instance created successfully")
        
        # Test basic configuration
        voice_shield.set_privacy_mode(PrivacyMode.PUBLIC)
        print("[SUCCESS] Privacy mode switching works")
        
        # Get performance stats
        stats = voice_shield.get_performance_stats()
        print(f"[SUCCESS] Performance stats: {stats}")
        
    except Exception as e:
        print(f"[FAILED] VoiceShield initialization failed: {e}")
        traceback.print_exc()
print()

# Test AI model components
print("--- AI Model Components ---")
test_import("core.ai_models.whisper_v3_processor", "Whisper-v3 Processor")
test_import("core.ai_models.styletts2_converter", "StyleTTS2 Converter")
test_import("core.ai_models.pyannote3_diarization", "Pyannote 3.0 Diarization")
test_import("core.ai_models.wavlm_emotion_processor", "WavLM Emotion Processor")
test_import("core.ai_models.audiocraft_inpainter", "AudioCraft Inpainter")
print()

# Test privacy technologies
print("--- Privacy Technologies ---")
test_import("core.privacy.privacy_enhancing_tech", "Privacy Enhancing Technologies")
print()

# Test TikTok integration
print("--- TikTok Integration ---")
test_import("integrations.tiktok.live_integration", "TikTok Live Integration")
print()

print("=== Test Complete ===")
