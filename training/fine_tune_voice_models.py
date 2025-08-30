"""
VoiceShield - Advanced Audio Model Fine-tuning with Real Voice Datasets
Specialized fine-tuning for voice privacy, emotion detection, and speaker recognition

This script focuses on:
1. Voice biometric anonymization models
2. Speaker diarization and recognition
3. Emotion detection and neutralization
4. Real-time voice conversion models
5. Audio quality enhancement for privacy
"""

import os
import sys
import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from datetime import datetime
import soundfile as sf
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AudioTrainingConfig:
    """Configuration for audio model training"""
    model_type: str
    sample_rate: int = 16000
    n_mels: int = 80
    hop_length: int = 256
    win_length: int = 1024
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class VoiceDatasetManager:
    """Advanced voice dataset management for privacy training"""
    
    def __init__(self, data_dir: str = "voice_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Voice dataset configurations
        self.voice_datasets = {
            'speaker_verification': [
                {
                    'name': 'voxceleb1',
                    'url': 'huggingface',
                    'dataset_id': 'facebook/voxceleb',
                    'description': 'VoxCeleb speaker verification dataset'
                },
                {
                    'name': 'common_voice_speakers',
                    'url': 'huggingface',
                    'dataset_id': 'mozilla-foundation/common_voice_16_1',
                    'description': 'Common Voice with speaker labels'
                }
            ],
            'emotion_recognition': [
                {
                    'name': 'crema_d',
                    'url': 'manual',
                    'description': 'CREMA-D emotion recognition dataset'
                },
                {
                    'name': 'savee',
                    'url': 'manual',
                    'description': 'SAVEE emotion recognition dataset'
                }
            ],
            'voice_conversion': [
                {
                    'name': 'vctk_corpus',
                    'url': 'manual',
                    'description': 'VCTK voice conversion corpus'
                }
            ]
        }
    
    async def download_voice_datasets(self):
        """Download and prepare voice datasets"""
        logger.info("🎵 Downloading voice datasets...")
        
        # Download Common Voice subset for testing
        await self._download_common_voice_subset()
        
        # Create synthetic voice data for privacy training
        await self._create_synthetic_voice_data()
        
        logger.info("✅ Voice datasets prepared!")
    
    async def _download_common_voice_subset(self):
        """Download subset of Common Voice for training"""
        try:
            from datasets import load_dataset
            
            logger.info("Downloading Common Voice subset...")
            dataset = load_dataset(
                "mozilla-foundation/common_voice_16_1", 
                "en", 
                split="train[:100]",
                trust_remote_code=True
            )
            
            # Save audio files
            cv_dir = self.data_dir / "common_voice"
            cv_dir.mkdir(exist_ok=True)
            
            audio_data = []
            for i, sample in enumerate(dataset):
                try:
                    # Extract audio
                    audio_array = sample['audio']['array']
                    sample_rate = sample['audio']['sampling_rate']
                    
                    # Save audio file
                    audio_path = cv_dir / f"audio_{i:04d}.wav"
                    sf.write(str(audio_path), audio_array, sample_rate)
                    
                    audio_data.append({
                        'file': str(audio_path),
                        'text': sample['sentence'],
                        'speaker': f"speaker_{hash(str(sample.get('client_id', i))) % 50}",
                        'duration': len(audio_array) / sample_rate
                    })
                    
                    if i >= 99:  # Limit for demo
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue
            
            # Save metadata
            metadata_path = cv_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(audio_data, f, indent=2)
            
            logger.info(f"✅ Common Voice: {len(audio_data)} samples prepared")
            
        except Exception as e:
            logger.error(f"Failed to download Common Voice: {e}")
    
    async def _create_synthetic_voice_data(self):
        """Create synthetic voice data for privacy training"""
        logger.info("🎨 Creating synthetic voice data...")
        
        synth_dir = self.data_dir / "synthetic"
        synth_dir.mkdir(exist_ok=True)
        
        # Create synthetic audio features for different privacy scenarios
        synthetic_data = []
        
        for speaker_id in range(10):
            for emotion in ['neutral', 'happy', 'sad', 'angry']:
                for privacy_level in ['public', 'private', 'sensitive']:
                    # Generate synthetic mel spectrogram features
                    mel_features = self._generate_synthetic_mel_spectrogram(
                        speaker_id, emotion, privacy_level
                    )
                    
                    # Create metadata
                    sample_data = {
                        'speaker_id': f"synth_speaker_{speaker_id}",
                        'emotion': emotion,
                        'privacy_level': privacy_level,
                        'features': mel_features.tolist(),
                        'duration': 3.0
                    }
                    
                    synthetic_data.append(sample_data)
        
        # Save synthetic data
        synth_metadata_path = synth_dir / "synthetic_metadata.json"
        with open(synth_metadata_path, 'w') as f:
            json.dump(synthetic_data, f, indent=2)
        
        logger.info(f"✅ Synthetic voice data: {len(synthetic_data)} samples created")
    
    def _generate_synthetic_mel_spectrogram(self, speaker_id: int, emotion: str, privacy_level: str) -> np.ndarray:
        """Generate synthetic mel spectrogram features"""
        
        # Create base features
        n_frames = 200  # ~3 seconds at 16kHz with hop_length=256
        n_mels = 80
        
        # Base frequency characteristics
        base_freq = 100 + speaker_id * 20  # Speaker-specific base frequency
        
        # Emotion modulation
        emotion_mods = {
            'neutral': 1.0,
            'happy': 1.2,
            'sad': 0.8,
            'angry': 1.4
        }
        emotion_mod = emotion_mods.get(emotion, 1.0)
        
        # Privacy level affects amplitude and frequency spread
        privacy_mods = {
            'public': {'amp': 1.0, 'spread': 1.0},
            'private': {'amp': 0.8, 'spread': 0.9},
            'sensitive': {'amp': 0.6, 'spread': 0.7}
        }
        privacy_mod = privacy_mods.get(privacy_level, {'amp': 1.0, 'spread': 1.0})
        
        # Generate mel spectrogram
        mel_spec = np.zeros((n_mels, n_frames))
        
        for mel_bin in range(n_mels):
            # Frequency content based on mel bin
            freq_weight = np.exp(-(mel_bin - 20) ** 2 / (2 * 10 ** 2))  # Peak around bin 20
            
            for frame in range(n_frames):
                # Temporal modulation
                temporal_mod = 0.8 + 0.4 * np.sin(2 * np.pi * frame / 50)
                
                # Combine all factors
                amplitude = (
                    freq_weight * 
                    emotion_mod * 
                    privacy_mod['amp'] * 
                    temporal_mod *
                    (0.5 + 0.5 * np.random.rand())  # Random variation
                )
                
                mel_spec[mel_bin, frame] = amplitude
        
        # Add noise and normalize
        mel_spec += 0.1 * np.random.randn(n_mels, n_frames)
        mel_spec = np.clip(mel_spec, 0, None)
        
        return mel_spec

class VoicePrivacyModel(nn.Module):
    """Neural network for voice privacy protection"""
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, output_dim: int = 80):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Encoder for voice features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Privacy transformation layers
        self.privacy_transformer = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Decoder for output features
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Ensure output is in valid range
        )
        
        # Privacy level embedding
        self.privacy_embedding = nn.Embedding(4, hidden_dim // 4)
        
    def forward(self, x, privacy_level):
        """Forward pass with privacy level conditioning"""
        batch_size, seq_len, _ = x.shape
        
        # Reshape for processing
        x_flat = x.view(-1, self.input_dim)
        
        # Encode
        encoded = self.encoder(x_flat)
        
        # Add privacy conditioning
        privacy_emb = self.privacy_embedding(privacy_level)
        privacy_emb = privacy_emb.unsqueeze(1).expand(-1, seq_len, -1)
        privacy_emb = privacy_emb.contiguous().view(-1, self.hidden_dim // 4)
        
        # Concatenate with encoded features
        combined = torch.cat([encoded, privacy_emb], dim=1)
        
        # Apply privacy transformation
        transformed = self.privacy_transformer(combined[:, :self.hidden_dim // 2])
        
        # Decode
        output = self.decoder(transformed)
        
        # Reshape back
        output = output.view(batch_size, seq_len, self.output_dim)
        
        return output

class SpeakerRecognitionModel(nn.Module):
    """Neural network for speaker recognition"""
    
    def __init__(self, input_dim: int = 80, num_speakers: int = 100, hidden_dim: int = 256):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_speakers)
        )
    
    def forward(self, x):
        """Forward pass for speaker classification"""
        # x shape: (batch, time, mels) -> (batch, mels, time)
        x = x.transpose(1, 2)
        
        # Extract features
        features = self.feature_extractor(x)  # (batch, 256, 1)
        features = features.squeeze(-1)  # (batch, 256)
        
        # Classify speaker
        output = self.classifier(features)
        
        return output

class EmotionRecognitionModel(nn.Module):
    """Neural network for emotion recognition"""
    
    def __init__(self, input_dim: int = 80, num_emotions: int = 4, hidden_dim: int = 256):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, num_emotions)
        )
    
    def forward(self, x):
        """Forward pass for emotion classification"""
        # x shape: (batch, time, mels) -> (batch, mels, time)
        x = x.transpose(1, 2)
        
        # Extract features
        features = self.feature_extractor(x)  # (batch, 256, 1)
        features = features.squeeze(-1)  # (batch, 256)
        
        # Classify emotion
        output = self.classifier(features)
        
        return output

class VoiceModelTrainer:
    """Comprehensive voice model trainer"""
    
    def __init__(self, config: AudioTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.privacy_model = VoicePrivacyModel().to(self.device)
        self.speaker_model = SpeakerRecognitionModel().to(self.device)
        self.emotion_model = EmotionRecognitionModel().to(self.device)
        
        # Initialize optimizers
        self.privacy_optimizer = torch.optim.Adam(self.privacy_model.parameters(), lr=config.learning_rate)
        self.speaker_optimizer = torch.optim.Adam(self.speaker_model.parameters(), lr=config.learning_rate)
        self.emotion_optimizer = torch.optim.Adam(self.emotion_model.parameters(), lr=config.learning_rate)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.models_dir = Path("fine_tuned_voice_models")
        self.models_dir.mkdir(exist_ok=True)
    
    async def train_all_models(self, dataset_manager: VoiceDatasetManager):
        """Train all voice models"""
        logger.info("🎵 Starting comprehensive voice model training...")
        
        # Prepare training data
        train_data = await self._prepare_training_data(dataset_manager)
        
        results = {}
        
        # Train privacy model
        logger.info("🔐 Training voice privacy model...")
        privacy_metrics = await self._train_privacy_model(train_data)
        results['voice_privacy'] = privacy_metrics
        
        # Train speaker recognition model
        logger.info("👤 Training speaker recognition model...")
        speaker_metrics = await self._train_speaker_model(train_data)
        results['speaker_recognition'] = speaker_metrics
        
        # Train emotion recognition model
        logger.info("😊 Training emotion recognition model...")
        emotion_metrics = await self._train_emotion_model(train_data)
        results['emotion_recognition'] = emotion_metrics
        
        # Save models
        await self._save_all_models()
        
        logger.info("✅ All voice models trained successfully!")
        return results
    
    async def _prepare_training_data(self, dataset_manager: VoiceDatasetManager) -> Dict:
        """Prepare training data for all models"""
        logger.info("📊 Preparing training data...")
        
        # Load synthetic data
        synth_dir = dataset_manager.data_dir / "synthetic"
        synth_metadata_path = synth_dir / "synthetic_metadata.json"
        
        if synth_metadata_path.exists():
            with open(synth_metadata_path, 'r') as f:
                synthetic_data = json.load(f)
        else:
            synthetic_data = []
        
        # Prepare data for each model
        training_data = {
            'privacy': [],
            'speaker': [],
            'emotion': []
        }
        
        emotion_labels = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3}
        privacy_labels = {'public': 0, 'private': 1, 'sensitive': 2}
        
        for sample in synthetic_data:
            mel_features = np.array(sample['features'])
            
            # Privacy training data
            privacy_level = privacy_labels[sample['privacy_level']]
            training_data['privacy'].append({
                'input': mel_features,
                'target': mel_features * 0.8,  # Reduced amplitude target
                'privacy_level': privacy_level
            })
            
            # Speaker training data
            speaker_id = int(sample['speaker_id'].split('_')[2])
            training_data['speaker'].append({
                'input': mel_features,
                'speaker_id': speaker_id
            })
            
            # Emotion training data
            emotion_id = emotion_labels[sample['emotion']]
            training_data['emotion'].append({
                'input': mel_features,
                'emotion_id': emotion_id
            })
        
        logger.info(f"Prepared training data:")
        logger.info(f"  - Privacy: {len(training_data['privacy'])} samples")
        logger.info(f"  - Speaker: {len(training_data['speaker'])} samples") 
        logger.info(f"  - Emotion: {len(training_data['emotion'])} samples")
        
        return training_data
    
    async def _train_privacy_model(self, train_data: Dict) -> Dict:
        """Train voice privacy protection model"""
        
        privacy_data = train_data['privacy']
        if not privacy_data:
            logger.warning("No privacy training data available")
            return {'loss': 0.0, 'accuracy': 0.0}
        
        self.privacy_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            # Create mini-batches
            batch_size = min(self.config.batch_size, len(privacy_data))
            
            for i in range(0, len(privacy_data), batch_size):
                batch = privacy_data[i:i + batch_size]
                
                # Prepare batch tensors
                inputs = []
                targets = []
                privacy_levels = []
                
                for sample in batch:
                    inputs.append(sample['input'])
                    targets.append(sample['target'])
                    privacy_levels.append(sample['privacy_level'])
                
                # Convert to tensors
                input_tensor = torch.FloatTensor(inputs).to(self.device)
                target_tensor = torch.FloatTensor(targets).to(self.device)
                privacy_tensor = torch.LongTensor(privacy_levels).to(self.device)
                
                # Forward pass
                self.privacy_optimizer.zero_grad()
                output = self.privacy_model(input_tensor, privacy_tensor)
                
                # Calculate loss
                loss = self.mse_loss(output, target_tensor)
                
                # Backward pass
                loss.backward()
                self.privacy_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
            
            if epoch % 5 == 0:
                logger.info(f"Privacy model epoch {epoch}, loss: {epoch_loss:.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': max(0.0, 1.0 - avg_loss),  # Approximation
            'model_size_mb': self._get_model_size(self.privacy_model),
            'inference_time_ms': await self._measure_inference_time(self.privacy_model)
        }
    
    async def _train_speaker_model(self, train_data: Dict) -> Dict:
        """Train speaker recognition model"""
        
        speaker_data = train_data['speaker']
        if not speaker_data:
            logger.warning("No speaker training data available")
            return {'loss': 0.0, 'accuracy': 0.0}
        
        self.speaker_model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            # Create mini-batches
            batch_size = min(self.config.batch_size, len(speaker_data))
            
            for i in range(0, len(speaker_data), batch_size):
                batch = speaker_data[i:i + batch_size]
                
                # Prepare batch tensors
                inputs = []
                speaker_ids = []
                
                for sample in batch:
                    inputs.append(sample['input'])
                    speaker_ids.append(sample['speaker_id'])
                
                # Convert to tensors
                input_tensor = torch.FloatTensor(inputs).to(self.device)
                speaker_tensor = torch.LongTensor(speaker_ids).to(self.device)
                
                # Forward pass
                self.speaker_optimizer.zero_grad()
                output = self.speaker_model(input_tensor)
                
                # Calculate loss
                loss = self.ce_loss(output, speaker_tensor)
                
                # Backward pass
                loss.backward()
                self.speaker_optimizer.step()
                
                # Calculate accuracy
                predicted = torch.argmax(output, dim=1)
                correct_predictions += (predicted == speaker_tensor).sum().item()
                total_predictions += len(batch)
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss
            
            if epoch % 5 == 0:
                accuracy = correct_predictions / max(total_predictions, 1)
                logger.info(f"Speaker model epoch {epoch}, loss: {epoch_loss:.4f}, accuracy: {accuracy:.4f}")
        
        accuracy = correct_predictions / max(total_predictions, 1)
        
        return {
            'loss': total_loss / max(epoch + 1, 1),
            'accuracy': accuracy,
            'model_size_mb': self._get_model_size(self.speaker_model),
            'inference_time_ms': await self._measure_inference_time(self.speaker_model)
        }
    
    async def _train_emotion_model(self, train_data: Dict) -> Dict:
        """Train emotion recognition model"""
        
        emotion_data = train_data['emotion']
        if not emotion_data:
            logger.warning("No emotion training data available")
            return {'loss': 0.0, 'accuracy': 0.0}
        
        self.emotion_model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            # Create mini-batches
            batch_size = min(self.config.batch_size, len(emotion_data))
            
            for i in range(0, len(emotion_data), batch_size):
                batch = emotion_data[i:i + batch_size]
                
                # Prepare batch tensors
                inputs = []
                emotion_ids = []
                
                for sample in batch:
                    inputs.append(sample['input'])
                    emotion_ids.append(sample['emotion_id'])
                
                # Convert to tensors
                input_tensor = torch.FloatTensor(inputs).to(self.device)
                emotion_tensor = torch.LongTensor(emotion_ids).to(self.device)
                
                # Forward pass
                self.emotion_optimizer.zero_grad()
                output = self.emotion_model(input_tensor)
                
                # Calculate loss
                loss = self.ce_loss(output, emotion_tensor)
                
                # Backward pass
                loss.backward()
                self.emotion_optimizer.step()
                
                # Calculate accuracy
                predicted = torch.argmax(output, dim=1)
                correct_predictions += (predicted == emotion_tensor).sum().item()
                total_predictions += len(batch)
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss
            
            if epoch % 5 == 0:
                accuracy = correct_predictions / max(total_predictions, 1)
                logger.info(f"Emotion model epoch {epoch}, loss: {epoch_loss:.4f}, accuracy: {accuracy:.4f}")
        
        accuracy = correct_predictions / max(total_predictions, 1)
        
        return {
            'loss': total_loss / max(epoch + 1, 1),
            'accuracy': accuracy,
            'model_size_mb': self._get_model_size(self.emotion_model),
            'inference_time_ms': await self._measure_inference_time(self.emotion_model)
        }
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    async def _measure_inference_time(self, model: nn.Module) -> float:
        """Measure model inference time"""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 200, 80).to(self.device)
        
        # Warm-up
        with torch.no_grad():
            if model == self.privacy_model:
                _ = model(dummy_input, torch.tensor([0]).to(self.device))
            else:
                _ = model(dummy_input)
        
        # Measure
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if model == self.privacy_model:
                _ = model(dummy_input, torch.tensor([0]).to(self.device))
            else:
                _ = model(dummy_input)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        return inference_time
    
    async def _save_all_models(self):
        """Save all trained models"""
        logger.info("💾 Saving voice models...")
        
        # Save privacy model
        privacy_path = self.models_dir / "voice_privacy_model.pth"
        torch.save(self.privacy_model.state_dict(), privacy_path)
        
        # Save speaker model
        speaker_path = self.models_dir / "speaker_recognition_model.pth"
        torch.save(self.speaker_model.state_dict(), speaker_path)
        
        # Save emotion model
        emotion_path = self.models_dir / "emotion_recognition_model.pth"
        torch.save(self.emotion_model.state_dict(), emotion_path)
        
        logger.info("✅ All voice models saved!")

async def main():
    """Main voice model training workflow"""
    logger.info("🎵 VoiceShield - Advanced Voice Model Fine-tuning Started")
    logger.info("=" * 80)
    
    try:
        # Initialize configurations
        config = AudioTrainingConfig(
            model_type='voice_privacy',
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=10
        )
        
        # Initialize dataset manager
        logger.info("📊 Initializing voice dataset manager...")
        dataset_manager = VoiceDatasetManager()
        
        # Download voice datasets
        logger.info("⬇️ Downloading voice datasets...")
        await dataset_manager.download_voice_datasets()
        
        # Initialize trainer
        logger.info("🔧 Initializing voice model trainer...")
        trainer = VoiceModelTrainer(config)
        
        # Train all voice models
        logger.info("🚀 Starting voice model training...")
        results = await trainer.train_all_models(dataset_manager)
        
        # Save results
        results_path = trainer.models_dir / "voice_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary
        logger.info("=" * 80)
        logger.info("🎉 Voice Model Training Complete!")
        logger.info(f"✅ Successfully trained {len(results)} voice models")
        
        for model_name, metrics in results.items():
            logger.info(f"   🎵 {model_name}: accuracy={metrics['accuracy']:.3f}, inference={metrics['inference_time_ms']:.1f}ms")
        
        logger.info("🎯 Voice models ready for real-time TikTok Live integration!")
        
    except Exception as e:
        logger.error(f"❌ Voice model training failed: {e}")
        raise

if __name__ == "__main__":
    # Run the voice model training
    asyncio.run(main())
