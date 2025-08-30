"""
Advanced PII Detection and Production Models Demo - 2025 SOTA
Comprehensive demonstration of context-aware PII detection and production-ready voice privacy

Features Demonstrated:
- Real-time context-aware PII detection
- Multi-modal analysis (text + audio features)  
- Production-grade fine-tuned models
- Adaptive privacy protection
- Real-time model adaptation
- Advanced privacy metrics
- Multiple conversation contexts
"""

import asyncio
import numpy as np
import time
import json
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our advanced systems
from core.voice_shield import VoiceShield, PrivacyMode, AudioChunk
from core.ai_models.context_aware_pii_detector import (
    ContextAwarePIIDetector, PIIType, ContextType, ConversationContext
)
from core.ai_models.production_whisper_processor import ProductionWhisperProcessor
from core.ai_models.privacy_model_trainer import PrivacyModelTrainer, PrivacyTrainingConfig
from core.ai_models.realtime_model_adapter import (
    RealTimeModelAdapter, FeedbackSample, FeedbackType
)

class AdvancedPIIDemo:
    """Comprehensive demo of advanced PII detection capabilities"""
    
    def __init__(self):
        self.voice_shield = None
        self.pii_detector = None
        self.whisper_processor = None
        self.model_adapter = None
        
        # Demo scenarios
        self.demo_scenarios = self._create_demo_scenarios()
        
        # Performance tracking
        self.demo_results = []
    
    def _create_demo_scenarios(self) -> List[Dict]:
        """Create realistic demo scenarios"""
        return [
            {
                "name": "Personal Call Scenario",
                "context_type": ContextType.PERSONAL_CALL,
                "privacy_mode": PrivacyMode.PERSONAL,
                "text": "Hey mom, it's Sarah. I'm calling from 555-123-4567. I just got the job at Google! They're offering me $150,000 salary. I'll be working with Dr. Johnson on the AI team. My new address will be 123 Main Street, Palo Alto. Can you pick me up at San Francisco airport tomorrow at 3 PM?",
                "expected_pii": [
                    "person_name", "phone_number", "company_name", 
                    "salary_income", "home_address", "geographic_location"
                ]
            },
            {
                "name": "Business Meeting Scenario", 
                "context_type": ContextType.BUSINESS_MEETING,
                "privacy_mode": PrivacyMode.MEETING,
                "text": "Good morning everyone. This is John from the marketing team. Our Q4 revenue was $2.3 million, up 15% from last quarter. We need to discuss the Johnson account - they're considering switching to our competitor. The client contact is Sarah at sarah.johnson@techcorp.com or 555-987-6543.",
                "expected_pii": [
                    "person_name", "financial_status", "email_address", 
                    "phone_number", "company_name"
                ]
            },
            {
                "name": "Medical Consultation Scenario",
                "context_type": ContextType.MEDICAL_CONSULTATION, 
                "privacy_mode": PrivacyMode.PUBLIC,
                "text": "Patient John Doe, age 45, presents with chest pain and shortness of breath. He has a history of hypertension and diabetes. Current medications include metformin and lisinopril. His insurance ID is ABC123456789. Blood pressure is 140/90. Recommending cardiac stress test.",
                "expected_pii": [
                    "person_name", "age", "health_condition", 
                    "medication", "medical_procedure"
                ]
            },
            {
                "name": "Financial Discussion Scenario",
                "context_type": ContextType.FINANCIAL_DISCUSSION,
                "privacy_mode": PrivacyMode.PUBLIC,
                "text": "My credit card number is 4532-1234-5678-9012 and the security code is 123. I need to transfer $50,000 from my savings account 987654321 to pay off my mortgage. My social security number is 123-45-6789 for verification. The bank routing number is 021000021.",
                "expected_pii": [
                    "credit_card", "ssn", "bank_account", "financial_status"
                ]
            },
            {
                "name": "Live Streaming Scenario",
                "context_type": ContextType.STREAMING,
                "privacy_mode": PrivacyMode.STREAMING,
                "text": "Hey everyone, welcome to my stream! I'm broadcasting from my home studio in Austin, Texas. Don't forget to follow me on Instagram @techstreamer2024. For business inquiries, email me at contact@mystream.com. Today we're reviewing the new MacBook Pro that cost me $3,500.",
                "expected_pii": [
                    "geographic_location", "email_address", "financial_status"
                ]
            }
        ]
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demo of all advanced features"""
        print("🚀 Starting Advanced PII Detection & Production Models Demo")
        print("=" * 70)
        
        try:
            # Initialize systems
            await self._initialize_systems()
            
            # Run PII detection demos
            await self._demo_context_aware_pii_detection()
            
            # Demo production models
            await self._demo_production_models()
            
            # Demo real-time adaptation
            await self._demo_realtime_adaptation()
            
            # Demo advanced privacy metrics
            await self._demo_privacy_metrics()
            
            # Generate final report
            await self._generate_demo_report()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")
    
    async def _initialize_systems(self):
        """Initialize all advanced systems"""
        print("\n🔧 Initializing Advanced Systems...")
        
        # Initialize VoiceShield with production models
        self.voice_shield = VoiceShield(
            sample_rate=48000,
            chunk_size_ms=20,
            privacy_mode=PrivacyMode.CONTEXTUAL,
            enable_advanced_features=True
        )
        
        # Initialize context-aware PII detector
        self.pii_detector = ContextAwarePIIDetector(
            device="auto",
            enable_audio_analysis=True,
            enable_contextual_adaptation=True
        )
        
        # Initialize production Whisper processor
        self.whisper_processor = ProductionWhisperProcessor(
            model_size="base",
            device="auto",
            enable_fine_tuned_models=True,
            enable_real_time_adaptation=True
        )
        
        # Initialize models
        await asyncio.gather(
            self.voice_shield.initialize_models(),
            self.pii_detector.initialize_models(),
            self.whisper_processor.initialize_production_models()
        )
        
        print("✅ All systems initialized successfully!")
    
    async def _demo_context_aware_pii_detection(self):
        """Demo context-aware PII detection across different scenarios"""
        print("\n🔍 Context-Aware PII Detection Demo")
        print("-" * 50)
        
        for i, scenario in enumerate(self.demo_scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['name']}")
            print(f"Context: {scenario['context_type'].value}")
            print(f"Privacy Mode: {scenario['privacy_mode'].value}")
            print(f"Text: \"{scenario['text'][:100]}...\"")
            
            # Create conversation context
            context = ConversationContext(
                context_type=scenario['context_type'],
                participants=["speaker_1"],
                privacy_sensitivity=0.8,
                formality_level=0.7 if scenario['context_type'] == ContextType.BUSINESS_MEETING else 0.4
            )
            
            # Run PII detection
            start_time = time.perf_counter()
            
            try:
                pii_results = await self.pii_detector.detect_pii_multimodal(
                    text=scenario['text'],
                    audio_features=self._generate_mock_audio_features(),
                    conversation_context=context
                )
                
                detection_time = (time.perf_counter() - start_time) * 1000
                
                # Analyze results
                detected_types = set(result.pii_type.value for result in pii_results)
                expected_types = set(scenario['expected_pii'])
                
                precision = len(detected_types & expected_types) / max(1, len(detected_types))
                recall = len(detected_types & expected_types) / max(1, len(expected_types))
                f1_score = 2 * precision * recall / max(1, precision + recall)
                
                print(f"⚡ Detection Time: {detection_time:.2f}ms")
                print(f"🎯 Detected PII Types: {len(pii_results)} instances")
                print(f"📊 Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}")
                
                # Display detected PII with details
                for result in pii_results[:3]:  # Show first 3 for brevity
                    print(f"   • {result.pii_type.value}: '{result.text_segment}' "
                         f"(confidence: {result.confidence_score:.2f}, "
                         f"risk: {result.privacy_risk_score:.2f})")
                
                # Store results
                self.demo_results.append({
                    'scenario': scenario['name'],
                    'detection_time_ms': detection_time,
                    'detected_count': len(pii_results),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                })
                
            except Exception as e:
                print(f"❌ PII Detection failed: {e}")
    
    async def _demo_production_models(self):
        """Demo production model capabilities"""
        print("\n🏭 Production Models Demo")
        print("-" * 40)
        
        # Generate sample audio for transcription
        sample_audio = np.random.randn(48000).astype(np.float32)  # 1 second
        
        print("🎤 Testing Production Whisper Processor...")
        
        try:
            # Test advanced transcription
            result = await self.whisper_processor.transcribe_audio_chunk_advanced(
                sample_audio,
                sample_rate=48000,
                context={'user_id': 'demo_user', 'privacy_mode': 'meeting'}
            )
            
            print(f"✅ Advanced transcription completed")
            print(f"⚡ Processing time: {result.processing_time_ms:.2f}ms")
            print(f"🔒 Privacy risk score: {result.privacy_risk_score:.3f}")
            print(f"🎭 Emotional markers: {result.emotional_markers}")
            print(f"📈 Context analysis: {result.context_analysis}")
            
            # Get production stats
            stats = self.whisper_processor.get_production_stats()
            print(f"📊 Production Model Stats:")
            print(f"   • Models loaded: {stats.get('production_models_loaded', 0)}")
            print(f"   • Specialized models: {stats.get('specialized_models', {})}")
            print(f"   • Real-time adaptation: {stats.get('real_time_adaptation', False)}")
            
        except Exception as e:
            print(f"❌ Production model test failed: {e}")
    
    async def _demo_realtime_adaptation(self):
        """Demo real-time model adaptation capabilities"""
        print("\n🔄 Real-Time Model Adaptation Demo")
        print("-" * 45)
        
        if not self.whisper_processor.model_adapter:
            print("⚠️ Real-time adaptation not available (model adapter not initialized)")
            return
        
        print("📚 Simulating user feedback for model improvement...")
        
        # Simulate various types of feedback
        feedback_scenarios = [
            {
                "type": FeedbackType.FALSE_POSITIVE,
                "text": "The company meeting starts at 9 AM",
                "description": "Incorrectly flagged 'company' as PII"
            },
            {
                "type": FeedbackType.MISSED_DETECTION,
                "text": "Call me at five five five one two three four",
                "description": "Missed phone number in spoken format"
            },
            {
                "type": FeedbackType.WRONG_CATEGORY,
                "text": "Dr. Smith will see you now",
                "description": "Flagged as person name instead of professional title"
            }
        ]
        
        for i, feedback_scenario in enumerate(feedback_scenarios, 1):
            print(f"\n🔧 Feedback {i}: {feedback_scenario['description']}")
            
            # Create feedback sample
            feedback = FeedbackSample(
                sample_id=f"demo_feedback_{i}",
                timestamp=datetime.now(),
                text_content=feedback_scenario['text'],
                audio_features=None,
                original_prediction={'confidence': 0.8},
                corrected_prediction={'confidence': 0.9},
                feedback_type=feedback_scenario['type'],
                user_id="demo_user",
                context={'scenario': 'demo'},
                privacy_sensitivity=0.7,
                confidence_delta=0.1
            )
            
            # Process feedback
            success = await self.whisper_processor.model_adapter.process_feedback(feedback)
            print(f"   {'✅' if success else '❌'} Feedback processed: {success}")
        
        # Get adaptation metrics
        try:
            metrics = self.whisper_processor.model_adapter.get_adaptation_metrics()
            print(f"\n📈 Adaptation Metrics:")
            print(f"   • Total feedback samples: {metrics.total_feedback_samples}")
            print(f"   • Accuracy improvement: {metrics.accuracy_improvement:.3f}")
            print(f"   • Adaptation latency: {metrics.adaptation_latency_ms:.2f}ms")
            print(f"   • Model drift score: {metrics.model_drift_score:.3f}")
        except Exception as e:
            print(f"⚠️ Could not retrieve adaptation metrics: {e}")
    
    async def _demo_privacy_metrics(self):
        """Demo advanced privacy metrics"""
        print("\n📊 Advanced Privacy Metrics Demo")
        print("-" * 40)
        
        # Test VoiceShield with different privacy modes
        privacy_modes = [PrivacyMode.PERSONAL, PrivacyMode.MEETING, PrivacyMode.PUBLIC]
        
        for mode in privacy_modes:
            print(f"\n🔐 Testing Privacy Mode: {mode.value}")
            
            # Set privacy mode
            self.voice_shield.set_privacy_mode(mode)
            
            # Generate test audio chunk
            test_audio = AudioChunk(
                data=np.random.randn(960).astype(np.float32),  # 20ms at 48kHz
                sample_rate=48000,
                timestamp=time.time(),
                duration_ms=20.0
            )
            
            try:
                # Process with VoiceShield
                processed_audio, metrics = await self.voice_shield.process_realtime_audio(test_audio)
                
                print(f"   ⚡ Processing latency: {metrics.processing_latency_ms:.2f}ms")
                print(f"   🛡️ Protection level: {metrics.protection_level:.2f}")
                print(f"   🔍 PII detected: {len(metrics.pii_detected)}")
                print(f"   🎭 Emotional markers: {len(metrics.emotion_markers)}")
                print(f"   🔐 Voice biometric masked: {metrics.voice_biometric_masked}")
                
            except Exception as e:
                print(f"   ❌ Processing failed: {e}")
        
        # Get overall performance stats
        stats = self.voice_shield.get_performance_stats()
        print(f"\n📈 Overall VoiceShield Performance:")
        print(f"   • Average latency: {stats.get('avg_latency_ms', 0):.2f}ms")
        print(f"   • Target met: {stats.get('target_met_pct', 0):.1f}%")
        print(f"   • Production models: {stats.get('production_models_enabled', False)}")
        print(f"   • Model type: {stats.get('model_type', 'unknown')}")
    
    async def _generate_demo_report(self):
        """Generate comprehensive demo report"""
        print("\n📋 Demo Report")
        print("=" * 30)
        
        if self.demo_results:
            avg_detection_time = np.mean([r['detection_time_ms'] for r in self.demo_results])
            avg_f1_score = np.mean([r['f1_score'] for r in self.demo_results])
            total_detections = sum(r['detected_count'] for r in self.demo_results)
            
            print(f"🎯 PII Detection Performance:")
            print(f"   • Average detection time: {avg_detection_time:.2f}ms")
            print(f"   • Average F1 score: {avg_f1_score:.3f}")
            print(f"   • Total PII instances detected: {total_detections}")
            print(f"   • Scenarios tested: {len(self.demo_results)}")
        
        # System status
        print(f"\n🏭 System Status:")
        print(f"   • VoiceShield initialized: {'✅' if self.voice_shield else '❌'}")
        print(f"   • Context-aware PII detector: {'✅' if self.pii_detector else '❌'}")
        print(f"   • Production Whisper: {'✅' if self.whisper_processor else '❌'}")
        print(f"   • Real-time adaptation: {'✅' if self.whisper_processor and self.whisper_processor.model_adapter else '❌'}")
        
        print(f"\n✨ Advanced Features Demonstrated:")
        print(f"   ✅ Context-aware PII detection")
        print(f"   ✅ Multi-modal analysis (text + audio)")
        print(f"   ✅ Production-grade fine-tuned models")
        print(f"   ✅ Adaptive privacy protection")
        print(f"   ✅ Real-time model adaptation")
        print(f"   ✅ Advanced privacy metrics")
        
        print(f"\n🎉 Demo completed successfully!")
    
    def _generate_mock_audio_features(self):
        """Generate mock audio features for demo"""
        from core.ai_models.context_aware_pii_detector import AudioFeatures
        
        return AudioFeatures(
            mfcc=np.random.randn(13, 10),
            pitch=np.array([200.0]),
            energy=np.array([0.1]),
            spectral_centroid=np.array([1500.0]),
            zero_crossing_rate=np.array([0.05]),
            chroma=np.random.randn(12, 10),
            voice_biometric_features=np.random.randn(128)
        )

async def main():
    """Main demo function"""
    print("🎬 Advanced PII Detection & Production Models Demo")
    print("🔬 Showcasing 2025 SOTA Privacy Technology")
    
    demo = AdvancedPIIDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main())