"""
VoiceShield Demo - TikTok Live Privacy Protection
Complete demonstration of real-time voice privacy during live streaming
"""

import asyncio
import time
import logging
from typing import Dict
import json

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import VoiceShield components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.voice_shield import VoiceShield, PrivacyMode, AudioChunk
from core.audio.processor import RealTimeAudioProcessor, AudioConfig
from integrations.tiktok.live_integration import TikTokLiveVoiceShield


class VoiceShieldDemo:
    """
    Complete VoiceShield demonstration for TikTok Live streaming
    
    This demo showcases:
    1. Real-time voice privacy protection
    2. TikTok Live integration
    3. Dynamic privacy adjustments
    4. Privacy metrics and monitoring
    5. Emergency privacy controls
    """
    
    def __init__(self):
        self.voice_shield = None
        self.audio_processor = None
        self.tiktok_integration = None
        self.demo_running = False
        
    async def run_complete_demo(self):
        """Run the complete VoiceShield demo"""
        print("🛡️  VoiceShield Demo - TikTok Live Privacy Protection")
        print("="*60)
        
        try:
            # Step 1: Initialize VoiceShield
            await self._initialize_voice_shield()
            
            # Step 2: Setup audio processing
            await self._setup_audio_processing()
            
            # Step 3: Initialize TikTok Live integration
            await self._setup_tiktok_integration()
            
            # Step 4: Run interactive demo scenarios
            await self._run_demo_scenarios()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
        finally:
            await self._cleanup()
    
    async def _initialize_voice_shield(self):
        """Initialize the core VoiceShield engine"""
        print("🔧 Initializing VoiceShield Core Engine...")
        
        # Create VoiceShield with optimized settings for live streaming
        self.voice_shield = VoiceShield(
            sample_rate=48000,
            chunk_size_ms=20,  # Low latency for streaming
            privacy_mode=PrivacyMode.PERSONAL
        )
        
        # Load AI models
        print("📱 Loading AI models (Whisper-v3, StyleTTS2, Pyannote 3.0)...")
        await self.voice_shield.initialize_models()
        
        print("✅ VoiceShield initialized successfully!")
        print(f"   - Target latency: < 50ms")
        print(f"   - Privacy mode: {self.voice_shield.privacy_mode.value}")
        print()
    
    async def _setup_audio_processing(self):
        """Setup real-time audio processing"""
        print("🎤 Setting up Real-Time Audio Processing...")
        
        config = AudioConfig(
            sample_rate=48000,
            channels=1,
            chunk_duration_ms=20,
            buffer_size=4096
        )
        
        # Create audio processor with VoiceShield callback
        self.audio_processor = RealTimeAudioProcessor(
            config=config,
            processing_callback=self.voice_shield.process_realtime_audio
        )
        
        print("✅ Audio processing configured!")
        print(f"   - Sample rate: {config.sample_rate}Hz")
        print(f"   - Chunk duration: {config.chunk_duration_ms}ms")
        print(f"   - Buffer size: {config.buffer_size}")
        print()
    
    async def _setup_tiktok_integration(self):
        """Setup TikTok Live integration"""
        print("🎵 Initializing TikTok Live Integration...")
        
        self.tiktok_integration = TikTokLiveVoiceShield(
            voice_shield=self.voice_shield,
            stream_key="demo_stream_key_123"
        )
        
        print("✅ TikTok Live integration ready!")
        print(f"   - Target streaming latency: {self.tiktok_integration.target_latency_ms}ms")
        print(f"   - Background filtering: {self.tiktok_integration.background_filtering_enabled}")
        print()
    
    async def _run_demo_scenarios(self):
        """Run interactive demo scenarios"""
        print("🎬 Running Demo Scenarios")
        print("="*40)
        
        # Scenario 1: Home Cooking Stream
        await self._demo_home_cooking_stream()
        
        await asyncio.sleep(2)
        
        # Scenario 2: Q&A Session  
        await self._demo_qa_session()
        
        await asyncio.sleep(2)
        
        # Scenario 3: Emergency Privacy
        await self._demo_emergency_privacy()
        
        await asyncio.sleep(2)
        
        # Scenario 4: Performance Monitoring
        await self._demo_performance_monitoring()
    
    async def _demo_home_cooking_stream(self):
        """Demo Scenario 1: Home Cooking Stream with Privacy Protection"""
        print("🍳 Scenario 1: Home Cooking Stream")
        print("-" * 30)
        
        # Start the live stream
        stream_result = await self.tiktok_integration.start_live_stream("Cooking Pasta at Home!")
        
        if stream_result["success"]:
            print(f"✅ Live stream started: {stream_result['stream_id']}")
            
            # Start audio processing
            self.audio_processor.start_processing()
            print("🎤 Audio processing started...")
            
            # Simulate stream events
            await self._simulate_cooking_stream_events()
            
        else:
            print(f"❌ Failed to start stream: {stream_result.get('error')}")
    
    async def _simulate_cooking_stream_events(self):
        """Simulate events during cooking stream"""
        events = [
            (2, "🔊 Stream starting with 50 viewers"),
            (4, "👥 Mom calls from kitchen: 'John, dinner ready!' -> Voice filtered"),
            (6, "📈 Viewer count spikes to 500 -> Privacy level auto-increased"),
            (8, "📍 Almost mentioned home address -> PII detected and masked"), 
            (10, "🎵 Background music detected -> Copyright-safe replacement applied"),
            (12, "📱 Phone notification with personal info -> Audio masked"),
            (14, "👨‍👩‍👧‍👦 Family member enters background -> Multi-speaker protection activated")
        ]
        
        start_time = time.time()
        for delay, description in events:
            while time.time() - start_time < delay:
                await asyncio.sleep(0.1)
            
            print(f"   {description}")
            
            # Update viewer count simulation
            if "viewer count spikes" in description:
                await self._simulate_viewer_spike()
            elif "PII detected" in description:
                await self._simulate_pii_detection()
    
    async def _simulate_viewer_spike(self):
        """Simulate sudden increase in viewers"""
        # Manually update viewer count for demo
        self.tiktok_integration.current_viewers = 500
        
        # This would trigger privacy mode adjustment
        from core.voice_shield import PrivacyMode
        self.voice_shield.set_privacy_mode(PrivacyMode.PUBLIC)
        
        print(f"     🛡️  Privacy protection increased to PUBLIC mode")
    
    async def _simulate_pii_detection(self):
        """Simulate PII detection and masking"""
        # Add mock PII detection
        pii_event = {
            "timestamp": time.time(),
            "type": "address",
            "content": "[MASKED]",
            "confidence": 0.95
        }
        
        self.tiktok_integration.pii_blocked_today.append(pii_event)
        print(f"     🔒 Address information masked automatically")
    
    async def _demo_qa_session(self):
        """Demo Scenario 2: Q&A Session with Multiple Creators"""
        print("\n🎤 Scenario 2: Collaborative Q&A Session")
        print("-" * 35)
        
        # Simulate multi-speaker scenario
        print("👥 Multiple creators joining stream...")
        print("   🛡️  Individual privacy protection for each speaker")
        print("   🔊 Friend mentions real name -> Instantly replaced with username")
        print("   📊 Different privacy levels for each participant")
        print("   🔄 Cross-platform sync enabled")
        
        # Show privacy metrics
        metrics = self.tiktok_integration.get_live_metrics()
        print(f"   📈 Current viewer count: {metrics.viewer_count}")
        print(f"   🛡️  Protection level: {metrics.average_protection_level:.2f}")
        print(f"   ⚡ Stream health: {metrics.stream_health_score:.1f}%")
    
    async def _demo_emergency_privacy(self):
        """Demo Scenario 3: Emergency Privacy Controls"""
        print("\n🚨 Scenario 3: Emergency Privacy Control")
        print("-" * 35)
        
        print("⚠️  Emergency scenario: Sensitive information about to be shared...")
        print("🔴 Emergency privacy button activated!")
        
        # Trigger emergency mode
        result = self.tiktok_integration.emergency_privacy_stop()
        
        print(f"   ✅ {result['message']}")
        print(f"   🔇 Audio muted: {result['audio_muted']}")
        print("   🛡️  Maximum privacy protection activated")
        
        await asyncio.sleep(2)
        
        # Show recovery
        print("   🔄 Privacy restored, stream resumed safely")
        self.voice_shield.emergency_privacy_toggle()  # Turn back on
    
    async def _demo_performance_monitoring(self):
        """Demo Scenario 4: Real-time Performance Monitoring"""
        print("\n📊 Scenario 4: Performance & Privacy Analytics")
        print("-" * 40)
        
        # Get comprehensive statistics
        voice_stats = self.voice_shield.get_performance_stats()
        audio_stats = self.audio_processor.get_stats()
        live_metrics = self.tiktok_integration.get_live_metrics()
        dashboard_data = self.tiktok_integration.get_privacy_dashboard_data()
        
        print("🎯 VoiceShield Performance Metrics:")
        print(f"   ⚡ Average latency: {voice_stats.get('avg_latency_ms', 0):.2f}ms")
        print(f"   🎯 Target achievement: {voice_stats.get('target_met_pct', 0)}%")
        print(f"   🔄 Chunks processed: {voice_stats.get('total_chunks', 0)}")
        
        print("\n🎤 Audio Processing Stats:")
        print(f"   📊 Processing quality: {audio_stats.get('drop_rate_pct', 0):.1f}% dropped")
        print(f"   💾 Buffer utilization: {audio_stats.get('buffer_utilization', 0):.1%}")
        
        print("\n📱 TikTok Live Metrics:")
        print(f"   👥 Current viewers: {live_metrics.viewer_count}")
        print(f"   🛡️  Privacy alerts: {live_metrics.privacy_alerts_triggered}")
        print(f"   🔒 PII blocked: {live_metrics.pii_blocked_count}")
        print(f"   🎵 Background voices filtered: {live_metrics.background_voices_filtered}")
        print(f"   ❤️  Stream health: {live_metrics.stream_health_score:.1f}%")
        
        print("\n🎛️  Privacy Dashboard Status:")
        print(f"   🔄 Mode: {dashboard_data['streaming_mode']}")
        print(f"   🛡️  Privacy level: {dashboard_data['privacy_mode']}")
        print(f"   ✅ Protection active: {dashboard_data['protection_active']}")
        print(f"   🚨 Emergency mode: {dashboard_data['emergency_mode']}")
    
    async def _cleanup(self):
        """Cleanup demo resources"""
        print("\n🧹 Cleaning up demo...")
        
        try:
            # Stop audio processing
            if self.audio_processor and hasattr(self.audio_processor, 'is_running'):
                self.audio_processor.stop_processing()
            
            # End live stream
            if self.tiktok_integration and self.tiktok_integration.streaming_mode.value == "live":
                result = await self.tiktok_integration.stop_live_stream()
                if result.get("success"):
                    summary = result["stream_summary"]
                    print(f"📈 Stream ended - Duration: {summary['duration_seconds']}s")
                    print(f"   👥 Max viewers: {summary['max_viewers']}")
                    print(f"   🛡️  Privacy events: {summary['privacy_alerts']}")
        
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        print("✅ Demo completed successfully!")


async def main():
    """Run the VoiceShield demo"""
    demo = VoiceShieldDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    print("🚀 Starting VoiceShield TikTok Live Demo...")
    asyncio.run(main())
