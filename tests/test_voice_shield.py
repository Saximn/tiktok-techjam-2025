"""
VoiceShield Test Suite
Comprehensive tests for voice privacy protection system
"""

import unittest
import asyncio
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.voice_shield import VoiceShield, PrivacyMode, AudioChunk, PrivacyMetrics
from core.audio.processor import AudioConfig, AudioBuffer, RealTimeAudioProcessor
from integrations.tiktok.live_integration import TikTokLiveVoiceShield, StreamingMode


class TestVoiceShieldCore(unittest.TestCase):
    """Test core VoiceShield functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.voice_shield = VoiceShield(
            sample_rate=48000,
            chunk_size_ms=20,
            privacy_mode=PrivacyMode.PERSONAL
        )
    
    def test_initialization(self):
        """Test VoiceShield initialization"""
        self.assertEqual(self.voice_shield.sample_rate, 48000)
        self.assertEqual(self.voice_shield.chunk_size_ms, 20)
        self.assertEqual(self.voice_shield.privacy_mode, PrivacyMode.PERSONAL)
        self.assertTrue(self.voice_shield.protection_active)
        
    def test_privacy_mode_changes(self):
        """Test privacy mode switching"""
        # Test mode changes
        self.voice_shield.set_privacy_mode(PrivacyMode.PUBLIC)
        self.assertEqual(self.voice_shield.privacy_mode, PrivacyMode.PUBLIC)
        
        self.voice_shield.set_privacy_mode(PrivacyMode.MEETING)
        self.assertEqual(self.voice_shield.privacy_mode, PrivacyMode.MEETING)
    
    def test_emergency_toggle(self):
        """Test emergency privacy toggle"""
        # Initially active
        self.assertTrue(self.voice_shield.protection_active)
        
        # Toggle off
        self.voice_shield.emergency_privacy_toggle()
        self.assertFalse(self.voice_shield.protection_active)
        
        # Toggle back on
        self.voice_shield.emergency_privacy_toggle()
        self.assertTrue(self.voice_shield.protection_active)
        self.assertEqual(self.voice_shield.privacy_mode, PrivacyMode.EMERGENCY)
    
    async def test_model_initialization(self):
        """Test AI model loading"""
        await self.voice_shield.initialize_models()
        
        # Check that all models are loaded
        self.assertIsNotNone(self.voice_shield.vad_model)
        self.assertIsNotNone(self.voice_shield.speaker_model)
        self.assertIsNotNone(self.voice_shield.style_transfer)
        self.assertIsNotNone(self.voice_shield.pii_detector)
        self.assertIsNotNone(self.voice_shield.emotion_neutralizer)
    
    async def test_audio_processing(self):
        """Test real-time audio processing"""
        await self.voice_shield.initialize_models()
        
        # Create test audio chunk
        audio_data = np.random.randn(960).astype(np.float32)  # 20ms at 48kHz
        test_chunk = AudioChunk(
            data=audio_data,
            sample_rate=48000,
            timestamp=time.time(),
            duration_ms=20
        )
        
        # Process audio
        processed_chunk, metrics = self.voice_shield.process_realtime_audio(test_chunk)
        
        # Verify processing
        self.assertIsInstance(processed_chunk, AudioChunk)
        self.assertIsInstance(metrics, PrivacyMetrics)
        self.assertEqual(processed_chunk.sample_rate, 48000)
        self.assertEqual(len(processed_chunk.data), len(audio_data))
        
        # Check metrics
        self.assertGreaterEqual(metrics.protection_level, 0)
        self.assertLessEqual(metrics.protection_level, 1)
        self.assertIsInstance(metrics.pii_detected, list)
        self.assertIsInstance(metrics.emotion_markers, list)
    
    def test_performance_stats(self):
        """Test performance statistics"""
        stats = self.voice_shield.get_performance_stats()
        
        # Should return no data initially
        self.assertIn("status", stats)


class TestAudioProcessing(unittest.TestCase):
    """Test audio processing components"""
    
    def setUp(self):
        """Setup audio processing tests"""
        self.config = AudioConfig(
            sample_rate=48000,
            channels=1,
            chunk_duration_ms=20
        )
    
    def test_audio_buffer(self):
        """Test circular audio buffer"""
        buffer = AudioBuffer(max_size_seconds=1.0, sample_rate=48000)
        
        # Test writing and reading
        test_data = np.random.randn(1000).astype(np.float32)
        buffer.write(test_data)
        
        # Read back data
        read_data = buffer.read(500)
        self.assertIsNotNone(read_data)
        self.assertEqual(len(read_data), 500)
        
        # Test insufficient data
        large_read = buffer.read(10000)
        self.assertIsNone(large_read)
    
    def test_audio_processor_creation(self):
        """Test audio processor creation"""
        def dummy_callback(chunk):
            return chunk, None
        
        processor = RealTimeAudioProcessor(
            config=self.config,
            processing_callback=dummy_callback
        )
        
        self.assertEqual(processor.config.sample_rate, 48000)
        self.assertEqual(processor.chunk_size, 960)  # 20ms at 48kHz
        self.assertFalse(processor.is_running)


class TestTikTokIntegration(unittest.TestCase):
    """Test TikTok Live integration"""
    
    def setUp(self):
        """Setup TikTok integration tests"""
        self.voice_shield = VoiceShield()
        self.tiktok = TikTokLiveVoiceShield(
            voice_shield=self.voice_shield,
            stream_key="test_key"
        )
    
    def test_initialization(self):
        """Test TikTok integration initialization"""
        self.assertEqual(self.tiktok.streaming_mode, StreamingMode.OFFLINE)
        self.assertEqual(self.tiktok.stream_key, "test_key")
        self.assertEqual(self.tiktok.target_latency_ms, 25)
        self.assertTrue(self.tiktok.audience_privacy_scaling)
    
    async def test_stream_lifecycle(self):
        """Test complete stream start/stop cycle"""
        # Start stream
        result = await self.tiktok.start_live_stream("Test Stream")
        self.assertTrue(result["success"])
        self.assertEqual(self.tiktok.streaming_mode, StreamingMode.LIVE)
        
        # Stop stream
        stop_result = await self.tiktok.stop_live_stream()
        self.assertTrue(stop_result["success"])
        self.assertEqual(self.tiktok.streaming_mode, StreamingMode.OFFLINE)
    
    def test_emergency_privacy(self):
        """Test emergency privacy controls"""
        result = self.tiktok.emergency_privacy_stop()
        
        self.assertEqual(result["status"], "emergency_activated")
        self.assertTrue(result["audio_muted"])
        self.assertTrue(self.tiktok.emergency_mode_active)
    
    def test_metrics_collection(self):
        """Test metrics and dashboard data"""
        metrics = self.tiktok.get_live_metrics()
        dashboard = self.tiktok.get_privacy_dashboard_data()
        
        # Check metrics structure
        self.assertGreaterEqual(metrics.viewer_count, 0)
        self.assertGreaterEqual(metrics.stream_duration_seconds, 0)
        self.assertIsInstance(metrics.privacy_alerts_triggered, int)
        
        # Check dashboard data
        self.assertIn("streaming_mode", dashboard)
        self.assertIn("privacy_mode", dashboard)
        self.assertIn("protection_active", dashboard)


class TestPrivacyFeatures(unittest.TestCase):
    """Test privacy protection features"""
    
    async def setUp(self):
        """Setup privacy feature tests"""
        self.voice_shield = VoiceShield()
        await self.voice_shield.initialize_models()
    
    async def test_pii_detection(self):
        """Test PII detection in audio"""
        # Create test audio (this would normally be speech with PII)
        audio_data = np.random.randn(960).astype(np.float32)
        
        # Test PII detector
        pii_detected = self.voice_shield.pii_detector.analyze(audio_data, [])
        self.assertIsInstance(pii_detected, list)
    
    async def test_emotion_detection(self):
        """Test emotion detection and neutralization"""
        # Create test audio
        audio_data = np.random.randn(960).astype(np.float32)
        
        # Test emotion detector
        emotions = self.voice_shield.emotion_neutralizer.detect(audio_data)
        self.assertIsInstance(emotions, list)
    
    async def test_voice_anonymization(self):
        """Test voice style transfer and anonymization"""
        audio_data = np.random.randn(960).astype(np.float32)
        speaker_segments = [{"speaker_id": "test", "start": 0, "end": 960}]
        
        # Test style transfer
        anonymized = self.voice_shield.style_transfer.anonymize(
            audio_data, 
            privacy_level=0.8,
            preserve_linguistic=True,
            speaker_segments=speaker_segments
        )
        
        self.assertEqual(len(anonymized), len(audio_data))
        self.assertIsInstance(anonymized, np.ndarray)


class TestPerformance(unittest.TestCase):
    """Test performance and latency requirements"""
    
    async def setUp(self):
        """Setup performance tests"""
        self.voice_shield = VoiceShield()
        await self.voice_shield.initialize_models()
    
    async def test_latency_requirements(self):
        """Test that processing meets latency requirements"""
        # Create test audio chunk
        audio_data = np.random.randn(960).astype(np.float32)
        test_chunk = AudioChunk(
            data=audio_data,
            sample_rate=48000,
            timestamp=time.time(),
            duration_ms=20
        )
        
        # Time processing
        start_time = time.perf_counter()
        processed_chunk, metrics = self.voice_shield.process_realtime_audio(test_chunk)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Should be under 50ms for real-time processing
        self.assertLess(processing_time_ms, 50.0, 
                       f"Processing took {processing_time_ms:.2f}ms, exceeds 50ms target")
        
        # Check that metrics report latency
        self.assertGreater(metrics.processing_latency_ms, 0)
    
    async def test_throughput(self):
        """Test processing throughput under load"""
        chunks_to_process = 100
        start_time = time.perf_counter()
        
        for i in range(chunks_to_process):
            audio_data = np.random.randn(960).astype(np.float32)
            test_chunk = AudioChunk(
                data=audio_data,
                sample_rate=48000,
                timestamp=time.time(),
                duration_ms=20
            )
            
            processed_chunk, metrics = self.voice_shield.process_realtime_audio(test_chunk)
        
        total_time = time.perf_counter() - start_time
        chunks_per_second = chunks_to_process / total_time
        
        # Should process at least 50 chunks per second (20ms each = real-time)
        self.assertGreater(chunks_per_second, 50, 
                          f"Processing rate {chunks_per_second:.1f} chunks/sec too slow")


# Async test runner
class AsyncTestRunner:
    """Helper to run async tests"""
    
    @staticmethod
    def run_async_test(test_func):
        """Run an async test function"""
        return asyncio.get_event_loop().run_until_complete(test_func)


if __name__ == "__main__":
    print("🧪 Running VoiceShield Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestVoiceShieldCore))
    suite.addTest(loader.loadTestsFromTestCase(TestAudioProcessing))
    suite.addTest(loader.loadTestsFromTestCase(TestTikTokIntegration))
    suite.addTest(loader.loadTestsFromTestCase(TestPrivacyFeatures))
    suite.addTest(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} error(s) occurred")
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
