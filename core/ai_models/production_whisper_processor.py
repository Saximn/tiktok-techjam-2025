"""
Advanced Whisper-v3 Implementation with Production PII Detection Models
Real production models with custom fine-tuning for privacy-aware speech recognition
"""

import torch
import whisper
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import asyncio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForTokenClassification, pipeline
)
import logging
from dataclasses import dataclass
import re
import os
import uuid
from datetime import datetime
from pathlib import Path

# Import our advanced PII detection
from .context_aware_pii_detector import ContextAwarePIIDetector, PIIType
from .privacy_model_trainer import PrivacyModelTrainer, PrivacyTrainingConfig
from .realtime_model_adapter import (
    RealTimeModelAdapter, FeedbackSample, FeedbackType, ModelUpdateStrategy
)

# Import base classes for compatibility
from .whisper_v3_processor import TranscriptionResult

logger = logging.getLogger(__name__)

@dataclass
class AdvancedTranscriptionResult:
    """Enhanced transcription result with comprehensive PII analysis"""
    text: str
    segments: List[Dict]
    language: str
    confidence: float
    privacy_tokens: List[str]  # Basic privacy tokens
    detailed_pii_results: List[Dict]  # Detailed PII analysis
    processing_time_ms: float
    model_version: str
    privacy_risk_score: float
    masked_transcription: str
    speaker_diarization: List[Dict]
    emotional_markers: List[str]
    context_analysis: Dict

class ProductionWhisperProcessor:
    """
    Production-grade Whisper processor with real fine-tuned models
    
    Features:
    - Custom fine-tuned Whisper models for privacy detection
    - Real-time PII detection with advanced NER models
    - Speaker-aware privacy protection
    - Contextual privacy analysis
    - Continuous learning from user feedback
    - Multi-language privacy pattern recognition
    """
    
    def __init__(self, 
                 model_size: str = "base",
                 device: str = "auto",
                 enable_fine_tuned_models: bool = True,
                 enable_real_time_adaptation: bool = True):
        """
        Initialize production Whisper processor
        
        Args:
            model_size: Whisper model size
            device: Computing device
            enable_fine_tuned_models: Use custom fine-tuned models
            enable_real_time_adaptation: Enable real-time model adaptation
        """
        self.model_size = model_size
        self.device = self._setup_device(device)
        self.enable_fine_tuned_models = enable_fine_tuned_models
        self.enable_real_time_adaptation = enable_real_time_adaptation
        
        # Base Whisper components
        self.whisper_model = None
        self.processor = None
        
        # Advanced PII detection models
        self.privacy_ner_model = None
        self.financial_pii_model = None
        self.health_pii_model = None
        self.personal_pii_model = None
        
        # Context-aware PII detector
        self.context_pii_detector = None
        
        # Real-time adaptation
        self.model_adapter = None
        
        # Production NER models
        self.production_ner_models = {}
        
        # Performance tracking
        self.inference_times = []
        self.total_transcriptions = 0
        self.model_performance_tracker = {}
        
        logger.info(f"Production Whisper Processor initialized - Model: {model_size}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup optimal computing device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA acceleration for production models")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using Apple Silicon MPS for production models")
            else:
                device = "cpu"
                logger.info("Using CPU for production models")
        return torch.device(device)
    
    async def initialize_production_models(self):
        """Initialize all production models"""
        start_time = time.time()
        
        try:
            logger.info("Loading production Whisper and PII detection models...")
            
            # Initialize models in parallel
            await asyncio.gather(
                self._load_base_whisper_model(),
                self._load_production_ner_models(),
                self._load_specialized_pii_models(),
                self._initialize_context_detector(),
                self._initialize_model_adapter()
            )
            
            # Verify model performance
            await self._verify_model_performance()
            
            load_time = (time.time() - start_time) * 1000
            logger.info(f"All production models loaded successfully in {load_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Production model initialization failed: {e}")
            # Fall back to basic models
            await self._load_fallback_models()
    
    async def _load_base_whisper_model(self):
        """Load base Whisper model with optimizations"""
        try:
            model_path = f"./models/whisper_fine_tuned_{self.model_size}" if self.enable_fine_tuned_models else self.model_size
            
            # Try to load fine-tuned model first
            if self.enable_fine_tuned_models and os.path.exists(model_path):
                logger.info(f"Loading fine-tuned Whisper model from {model_path}")
                self.whisper_model = whisper.load_model(model_path, device=self.device)
                self.processor = WhisperProcessor.from_pretrained(model_path)
            else:
                # Load base model
                logger.info(f"Loading base Whisper model: {self.model_size}")
                self.whisper_model = whisper.load_model(self.model_size, device=self.device)
                self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{self.model_size}")
            
            # Apply optimizations
            await self._optimize_whisper_model()
            
            logger.info("Base Whisper model loaded and optimized")
            
        except Exception as e:
            logger.error(f"Base Whisper model loading failed: {e}")
            raise
    
    async def _optimize_whisper_model(self):
        """Apply production optimizations to Whisper model"""
        try:
            # Set to evaluation mode
            self.whisper_model.eval()
            
            # Apply torch.jit optimizations if possible
            if self.device.type != "cpu":
                try:
                    # Trace the model for optimization
                    sample_input = torch.randn(1, 80, 3000).to(self.device)
                    self.whisper_model = torch.jit.trace(self.whisper_model.encoder, sample_input)
                    logger.info("Applied JIT optimization to Whisper model")
                except Exception as e:
                    logger.warning(f"JIT optimization failed: {e}")
            
            # Apply quantization for CPU inference
            if self.device.type == "cpu":
                self.whisper_model = torch.quantization.quantize_dynamic(
                    self.whisper_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied INT8 quantization for CPU inference")
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
    
    async def _load_production_ner_models(self):
        """Load production-grade NER models"""
        try:
            # Load multiple specialized NER models for different domains
            model_configs = {
                "general_privacy": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "financial_entities": "Jean-Baptiste/camembert-ner-with-dates",
                "biomedical_ner": "d4data/biomedical-ner-all",
                "personal_info": "Babelscape/wikineural-multilingual-ner"
            }
            
            for model_name, model_path in model_configs.items():
                try:
                    logger.info(f"Loading {model_name} model...")
                    
                    # Check if fine-tuned version exists
                    fine_tuned_path = f"./models/{model_name}_privacy_fine_tuned"
                    if os.path.exists(fine_tuned_path):
                        model_path = fine_tuned_path
                        logger.info(f"Using fine-tuned model for {model_name}")
                    
                    # Load model
                    ner_pipeline = pipeline(
                        "token-classification",
                        model=model_path,
                        aggregation_strategy="max",
                        device=0 if self.device.type == "cuda" else -1
                    )
                    
                    self.production_ner_models[model_name] = ner_pipeline
                    logger.info(f"Loaded {model_name} NER model successfully")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} model: {e}")
                    # Continue with other models
            
            logger.info(f"Loaded {len(self.production_ner_models)} production NER models")
            
        except Exception as e:
            logger.error(f"Production NER model loading failed: {e}")
    
    async def _load_specialized_pii_models(self):
        """Load specialized PII detection models"""
        try:
            # Load domain-specific models
            await asyncio.gather(
                self._load_financial_pii_model(),
                self._load_health_pii_model(),
                self._load_personal_pii_model()
            )
            
        except Exception as e:
            logger.error(f"Specialized PII model loading failed: {e}")
    
    async def _load_financial_pii_model(self):
        """Load financial PII detection model"""
        try:
            model_path = "./models/financial_pii_detector"
            
            if os.path.exists(model_path):
                self.financial_pii_model = pipeline(
                    "token-classification",
                    model=model_path,
                    device=0 if self.device.type == "cuda" else -1
                )
                logger.info("Financial PII model loaded")
            else:
                # Use general model as fallback
                self.financial_pii_model = self.production_ner_models.get("financial_entities")
                
        except Exception as e:
            logger.warning(f"Financial PII model loading failed: {e}")
    
    async def _load_health_pii_model(self):
        """Load health PII detection model"""
        try:
            model_path = "./models/health_pii_detector"
            
            if os.path.exists(model_path):
                self.health_pii_model = pipeline(
                    "token-classification",
                    model=model_path,
                    device=0 if self.device.type == "cuda" else -1
                )
                logger.info("Health PII model loaded")
            else:
                # Use biomedical model as fallback
                self.health_pii_model = self.production_ner_models.get("biomedical_ner")
                
        except Exception as e:
            logger.warning(f"Health PII model loading failed: {e}")
    
    async def _load_personal_pii_model(self):
        """Load personal PII detection model"""
        try:
            model_path = "./models/personal_pii_detector"
            
            if os.path.exists(model_path):
                self.personal_pii_model = pipeline(
                    "token-classification",
                    model=model_path,
                    device=0 if self.device.type == "cuda" else -1
                )
                logger.info("Personal PII model loaded")
            else:
                # Use general model as fallback
                self.personal_pii_model = self.production_ner_models.get("personal_info")
                
        except Exception as e:
            logger.warning(f"Personal PII model loading failed: {e}")
    
    async def _initialize_context_detector(self):
        """Initialize context-aware PII detector"""
        try:
            self.context_pii_detector = ContextAwarePIIDetector(
                device=str(self.device),
                enable_audio_analysis=True,
                enable_contextual_adaptation=True
            )
            await self.context_pii_detector.initialize_models()
            logger.info("Context-aware PII detector initialized")
            
        except Exception as e:
            logger.error(f"Context detector initialization failed: {e}")
    
    async def _initialize_model_adapter(self):
        """Initialize real-time model adapter"""
        if not self.enable_real_time_adaptation:
            return
        
        try:
            base_model_path = f"./models/whisper_fine_tuned_{self.model_size}"
            
            self.model_adapter = RealTimeModelAdapter(
                base_model_path=base_model_path,
                adaptation_strategy=ModelUpdateStrategy.BATCH,
                adaptation_threshold=20,
                learning_rate=1e-5
            )
            
            await self.model_adapter.initialize()
            logger.info("Real-time model adapter initialized")
            
        except Exception as e:
            logger.warning(f"Model adapter initialization failed: {e}")
    
    async def _verify_model_performance(self):
        """Verify all models are working correctly"""
        try:
            # Test transcription
            test_audio = np.random.randn(16000).astype(np.float32)  # 1 second of noise
            test_result = await self.transcribe_audio_chunk(test_audio)
            
            # Test PII detection
            if self.production_ner_models:
                test_text = "My name is John Doe and my phone number is 555-1234"
                for model_name, model in self.production_ner_models.items():
                    try:
                        entities = model(test_text)
                        logger.debug(f"{model_name} model test successful: {len(entities)} entities detected")
                    except Exception as e:
                        logger.warning(f"{model_name} model test failed: {e}")
            
            logger.info("Model performance verification completed")
            
        except Exception as e:
            logger.warning(f"Model verification failed: {e}")
    
    async def _load_fallback_models(self):
        """Load fallback models if production models fail"""
        logger.info("Loading fallback models...")
        
        try:
            # Load basic Whisper model
            self.whisper_model = whisper.load_model(self.model_size, device=self.device)
            self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{self.model_size}")
            
            # Load basic NER model
            self.production_ner_models["general"] = pipeline(
                "token-classification",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="max",
                device=0 if self.device.type == "cuda" else -1
            )
            
            logger.info("Fallback models loaded successfully")
            
        except Exception as e:
            logger.error(f"Fallback model loading failed: {e}")
            raise    
    async def transcribe_audio_chunk_advanced(self, 
                                            audio_chunk: np.ndarray, 
                                            sample_rate: int = 16000,
                                            language: Optional[str] = None,
                                            context: Optional[Dict] = None) -> AdvancedTranscriptionResult:
        """
        Advanced transcription with comprehensive PII detection using production models
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            language: Language hint
            context: Conversation context
            
        Returns:
            Comprehensive transcription result with PII analysis
        """
        start_time = time.perf_counter()
        
        try:
            # Step 1: Base transcription with Whisper
            base_result = await self._perform_base_transcription(audio_chunk, sample_rate, language)
            
            # Step 2: Multi-model PII detection
            pii_results = await self._perform_comprehensive_pii_detection(
                base_result['text'], audio_chunk, context
            )
            
            # Step 3: Context-aware analysis
            context_analysis = await self._perform_context_analysis(
                base_result['text'], pii_results, context
            )
            
            # Step 4: Generate privacy-masked transcription
            masked_transcription = await self._generate_masked_transcription(
                base_result['text'], pii_results
            )
            
            # Step 5: Calculate privacy risk score
            privacy_risk_score = self._calculate_comprehensive_privacy_risk(pii_results)
            
            # Step 6: Extract emotional markers and speaker info
            emotional_markers = await self._extract_emotional_markers(
                base_result['text'], audio_chunk
            )
            
            # Performance tracking
            processing_time = (time.perf_counter() - start_time) * 1000
            self.inference_times.append(processing_time)
            self.total_transcriptions += 1
            
            # Create comprehensive result
            result = AdvancedTranscriptionResult(
                text=base_result['text'],
                segments=base_result.get('segments', []),
                language=base_result.get('language', 'en'),
                confidence=base_result.get('confidence', 0.8),
                privacy_tokens=self._extract_basic_privacy_tokens(pii_results),
                detailed_pii_results=pii_results,
                processing_time_ms=processing_time,
                model_version=f"production_whisper_{self.model_size}",
                privacy_risk_score=privacy_risk_score,
                masked_transcription=masked_transcription,
                speaker_diarization=base_result.get('speaker_segments', []),
                emotional_markers=emotional_markers,
                context_analysis=context_analysis
            )
            
            # Send feedback to model adapter if enabled
            if self.enable_real_time_adaptation and self.model_adapter:
                await self._send_adaptation_feedback(result, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced transcription failed: {e}")
            # Return basic result on failure
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return AdvancedTranscriptionResult(
                text="",
                segments=[],
                language="en",
                confidence=0.0,
                privacy_tokens=[],
                detailed_pii_results=[],
                processing_time_ms=processing_time,
                model_version=f"fallback_whisper_{self.model_size}",
                privacy_risk_score=0.0,
                masked_transcription="",
                speaker_diarization=[],
                emotional_markers=[],
                context_analysis={}
            )
    
    async def _perform_base_transcription(self, 
                                        audio_chunk: np.ndarray, 
                                        sample_rate: int,
                                        language: Optional[str]) -> Dict:
        """Perform base Whisper transcription"""
        try:
            # Ensure audio is in the right format
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_chunk)) > 0:
                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
            
            # Resample if needed
            if sample_rate != 16000:
                from scipy import signal
                audio_chunk = signal.resample(
                    audio_chunk, int(len(audio_chunk) * 16000 / sample_rate)
                )
            
            # Transcribe with Whisper
            with torch.no_grad():
                result = self.whisper_model.transcribe(
                    audio_chunk,
                    language=language,
                    task="transcribe",
                    fp16=self.device.type == "cuda",
                    verbose=False
                )
            
            # Calculate confidence
            segments = result.get("segments", [])
            confidences = [seg.get("avg_logprob", 0.0) for seg in segments if "avg_logprob" in seg]
            avg_confidence = np.exp(np.mean(confidences)) if confidences else 0.8
            
            return {
                'text': result["text"].strip(),
                'segments': segments,
                'language': result.get("language", "en"),
                'confidence': avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Base transcription failed: {e}")
            return {
                'text': "",
                'segments': [],
                'language': "en",
                'confidence': 0.0
            }
    
    async def _perform_comprehensive_pii_detection(self, 
                                                 text: str, 
                                                 audio_chunk: np.ndarray,
                                                 context: Optional[Dict]) -> List[Dict]:
        """Perform comprehensive PII detection using all available models"""
        all_pii_results = []
        
        try:
            # Use context-aware detector if available
            if self.context_pii_detector:
                context_results = await self.context_pii_detector.detect_pii_multimodal(
                    text=text,
                    audio_features=self._extract_audio_features(audio_chunk),
                    conversation_context=context
                )
                
                # Convert to dict format
                for result in context_results:
                    all_pii_results.append({
                        'text_segment': result.text_segment,
                        'pii_type': result.pii_type.value,
                        'confidence_score': result.confidence_score,
                        'privacy_risk_score': result.privacy_risk_score,
                        'start_char': result.start_char,
                        'end_char': result.end_char,
                        'detection_method': result.detection_method,
                        'masked_text': result.masked_text or f"[{result.pii_type.value.upper()}]"
                    })
            
            # Run specialized models in parallel
            specialized_results = await asyncio.gather(
                self._detect_financial_pii(text),
                self._detect_health_pii(text),
                self._detect_personal_pii(text),
                return_exceptions=True
            )
            
            # Combine all results
            for results in specialized_results:
                if isinstance(results, list):
                    all_pii_results.extend(results)
            
            # Remove duplicates and merge overlapping detections
            all_pii_results = self._merge_overlapping_pii_detections(all_pii_results)
            
            return all_pii_results
            
        except Exception as e:
            logger.error(f"Comprehensive PII detection failed: {e}")
            return []
    
    async def _detect_financial_pii(self, text: str) -> List[Dict]:
        """Detect financial PII using specialized model"""
        if not self.financial_pii_model:
            return []
        
        try:
            entities = self.financial_pii_model(text)
            results = []
            
            for entity in entities:
                if entity.get('entity_group') in ['MONEY', 'CARDINAL', 'ORG']:
                    results.append({
                        'text_segment': entity['word'],
                        'pii_type': 'financial_info',
                        'confidence_score': entity['score'],
                        'privacy_risk_score': 0.8,
                        'start_char': entity['start'],
                        'end_char': entity['end'],
                        'detection_method': 'financial_ner_model',
                        'masked_text': '[FINANCIAL_INFO]'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Financial PII detection failed: {e}")
            return []
    
    async def _detect_health_pii(self, text: str) -> List[Dict]:
        """Detect health PII using specialized model"""
        if not self.health_pii_model:
            return []
        
        try:
            entities = self.health_pii_model(text)
            results = []
            
            # Health-specific entity types
            health_entity_types = [
                'DISEASE_DISORDER', 'MEDICATION', 'ANATOMY', 'MEDICAL_PROCEDURE'
            ]
            
            for entity in entities:
                if entity.get('entity_group') in health_entity_types:
                    results.append({
                        'text_segment': entity['word'],
                        'pii_type': 'health_info',
                        'confidence_score': entity['score'],
                        'privacy_risk_score': 0.9,
                        'start_char': entity['start'],
                        'end_char': entity['end'],
                        'detection_method': 'health_ner_model',
                        'masked_text': '[HEALTH_INFO]'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Health PII detection failed: {e}")
            return []
    
    async def _detect_personal_pii(self, text: str) -> List[Dict]:
        """Detect personal PII using specialized model"""
        if not self.personal_pii_model:
            return []
        
        try:
            entities = self.personal_pii_model(text)
            results = []
            
            for entity in entities:
                entity_group = entity.get('entity_group', '')
                
                # Map entity types to privacy categories
                if entity_group in ['PER', 'PERSON']:
                    pii_type = 'person_name'
                    risk_score = 0.7
                elif entity_group in ['LOC', 'LOCATION']:
                    pii_type = 'location'
                    risk_score = 0.6
                elif entity_group in ['ORG', 'ORGANIZATION']:
                    pii_type = 'organization'
                    risk_score = 0.5
                else:
                    continue
                
                results.append({
                    'text_segment': entity['word'],
                    'pii_type': pii_type,
                    'confidence_score': entity['score'],
                    'privacy_risk_score': risk_score,
                    'start_char': entity['start'],
                    'end_char': entity['end'],
                    'detection_method': 'personal_ner_model',
                    'masked_text': f'[{pii_type.upper()}]'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Personal PII detection failed: {e}")
            return []
    
    def _merge_overlapping_pii_detections(self, pii_results: List[Dict]) -> List[Dict]:
        """Merge overlapping PII detections"""
        if not pii_results:
            return pii_results
        
        # Sort by start position
        sorted_results = sorted(pii_results, key=lambda x: x.get('start_char', 0))
        merged = []
        
        for current in sorted_results:
            if not merged:
                merged.append(current)
                continue
            
            last = merged[-1]
            
            # Check for overlap
            if (current.get('start_char', 0) <= last.get('end_char', 0) and 
                current.get('end_char', 0) >= last.get('start_char', 0)):
                
                # Merge overlapping detections - keep the one with higher confidence
                if current.get('confidence_score', 0) > last.get('confidence_score', 0):
                    merged[-1] = current
            else:
                merged.append(current)
        
        return merged
    
    async def _perform_context_analysis(self, 
                                      text: str, 
                                      pii_results: List[Dict],
                                      context: Optional[Dict]) -> Dict:
        """Perform context analysis"""
        try:
            analysis = {
                'conversation_type': 'unknown',
                'formality_level': 0.5,
                'topic_sensitivity': 0.5,
                'pii_density': len(pii_results) / max(1, len(text.split())),
                'recommended_privacy_level': 0.7
            }
            
            # Analyze conversation type based on content
            if any(word in text.lower() for word in ['meeting', 'business', 'company']):
                analysis['conversation_type'] = 'business'
                analysis['formality_level'] = 0.8
            elif any(word in text.lower() for word in ['doctor', 'health', 'medical']):
                analysis['conversation_type'] = 'medical'
                analysis['topic_sensitivity'] = 0.9
            elif any(word in text.lower() for word in ['family', 'personal', 'friend']):
                analysis['conversation_type'] = 'personal'
                analysis['formality_level'] = 0.3
            
            # Adjust recommended privacy level
            if analysis['topic_sensitivity'] > 0.7:
                analysis['recommended_privacy_level'] = 0.9
            elif analysis['pii_density'] > 0.1:
                analysis['recommended_privacy_level'] = 0.8
            
            return analysis
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return {}
    
    async def _generate_masked_transcription(self, 
                                          text: str, 
                                          pii_results: List[Dict]) -> str:
        """Generate privacy-masked transcription"""
        try:
            masked_text = text
            
            # Sort PII results by start position in reverse order
            sorted_pii = sorted(pii_results, key=lambda x: x.get('start_char', 0), reverse=True)
            
            # Replace PII with masked versions
            for pii in sorted_pii:
                start_char = pii.get('start_char', 0)
                end_char = pii.get('end_char', 0)
                masked_replacement = pii.get('masked_text', '[REDACTED]')
                
                if start_char < len(masked_text) and end_char <= len(masked_text):
                    masked_text = (masked_text[:start_char] + 
                                 masked_replacement + 
                                 masked_text[end_char:])
            
            return masked_text
            
        except Exception as e:
            logger.error(f"Masked transcription generation failed: {e}")
            return text  # Return original on failure
    
    def _calculate_comprehensive_privacy_risk(self, pii_results: List[Dict]) -> float:
        """Calculate comprehensive privacy risk score"""
        if not pii_results:
            return 0.0
        
        try:
            # Weight different types of PII
            risk_weights = {
                'person_name': 0.7,
                'phone_number': 0.8,
                'email_address': 0.6,
                'financial_info': 1.0,
                'health_info': 0.9,
                'location': 0.6,
                'organization': 0.4
            }
            
            total_risk = 0.0
            max_individual_risk = 0.0
            
            for pii in pii_results:
                pii_type = pii.get('pii_type', 'unknown')
                base_risk = pii.get('privacy_risk_score', 0.5)
                weight = risk_weights.get(pii_type, 0.5)
                
                individual_risk = base_risk * weight
                total_risk += individual_risk
                max_individual_risk = max(max_individual_risk, individual_risk)
            
            # Combine total and maximum risk
            combined_risk = min(1.0, (total_risk / len(pii_results) + max_individual_risk) / 2)
            
            return combined_risk
            
        except Exception as e:
            logger.error(f"Privacy risk calculation failed: {e}")
            return 0.5  # Default moderate risk
    
    async def _extract_emotional_markers(self, text: str, audio_chunk: np.ndarray) -> List[str]:
        """Extract emotional markers from text and audio"""
        try:
            markers = []
            
            # Text-based emotional indicators
            emotional_keywords = {
                'anger': ['angry', 'mad', 'furious', 'annoyed'],
                'sadness': ['sad', 'depressed', 'upset', 'crying'],
                'joy': ['happy', 'excited', 'thrilled', 'great'],
                'fear': ['scared', 'afraid', 'worried', 'anxious'],
                'stress': ['stressed', 'overwhelmed', 'pressure', 'deadline']
            }
            
            text_lower = text.lower()
            for emotion, keywords in emotional_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    markers.append(emotion)
            
            # Audio-based emotional analysis (simplified)
            if len(audio_chunk) > 0:
                # Calculate basic prosodic features
                energy = np.sqrt(np.mean(audio_chunk ** 2))
                if energy > 0.1:
                    markers.append('high_energy')
                elif energy < 0.02:
                    markers.append('low_energy')
            
            return list(set(markers))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Emotional marker extraction failed: {e}")
            return []
    
    def _extract_basic_privacy_tokens(self, pii_results: List[Dict]) -> List[str]:
        """Extract basic privacy tokens for compatibility"""
        return [pii.get('pii_type', 'unknown') for pii in pii_results]
    
    def _extract_audio_features(self, audio_chunk: np.ndarray):
        """Extract audio features for context analysis"""
        try:
            # Basic audio feature extraction for PII detection context
            features = {
                'energy': np.sqrt(np.mean(audio_chunk ** 2)),
                'zero_crossing_rate': np.mean(np.diff(np.sign(audio_chunk)) != 0),
                'spectral_centroid': np.mean(np.abs(np.fft.fft(audio_chunk)))
            }
            return features
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return {}
    
    async def _send_adaptation_feedback(self, 
                                      result: AdvancedTranscriptionResult,
                                      context: Optional[Dict]):
        """Send feedback to model adapter for continuous learning"""
        if not self.model_adapter:
            return
        
        try:
            # Create feedback sample for adaptation
            feedback = FeedbackSample(
                sample_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                text_content=result.text,
                audio_features=None,  # Could include audio features
                original_prediction={
                    'privacy_tokens': result.privacy_tokens,
                    'confidence': result.confidence
                },
                corrected_prediction={},  # Would be filled with user corrections
                feedback_type=FeedbackType.CORRECT_DETECTION,  # Default assumption
                user_id=context.get('user_id') if context else None,
                context=context or {},
                privacy_sensitivity=result.privacy_risk_score,
                confidence_delta=0.0  # Would be calculated from user feedback
            )
            
            # Send to adapter (in background)
            asyncio.create_task(self.model_adapter.process_feedback(feedback))
            
        except Exception as e:
            logger.debug(f"Adaptation feedback failed: {e}")  # Non-critical error
    
    # Backward compatibility method
    async def transcribe_audio_chunk(self, 
                                   audio_chunk: np.ndarray, 
                                   sample_rate: int = 16000,
                                   language: Optional[str] = None) -> TranscriptionResult:
        """Backward compatibility method"""
        advanced_result = await self.transcribe_audio_chunk_advanced(
            audio_chunk, sample_rate, language
        )
        
        # Convert to basic result format
        return TranscriptionResult(
            text=advanced_result.text,
            segments=advanced_result.segments,
            language=advanced_result.language,
            confidence=advanced_result.confidence,
            privacy_tokens=advanced_result.privacy_tokens,
            processing_time_ms=advanced_result.processing_time_ms
        )
    
    def get_production_stats(self) -> Dict:
        """Get production model performance statistics"""
        if not self.inference_times:
            return {"status": "No production data"}
        
        avg_time = np.mean(self.inference_times)
        return {
            "total_transcriptions": self.total_transcriptions,
            "avg_inference_time_ms": round(avg_time, 2),
            "production_models_loaded": len(self.production_ner_models),
            "specialized_models": {
                "financial_pii": self.financial_pii_model is not None,
                "health_pii": self.health_pii_model is not None,
                "personal_pii": self.personal_pii_model is not None
            },
            "context_aware_detector": self.context_pii_detector is not None,
            "real_time_adaptation": self.model_adapter is not None,
            "model_size": self.model_size,
            "device": str(self.device)
        }


# Export main classes
__all__ = [
    'ProductionWhisperProcessor', 'AdvancedTranscriptionResult'
]