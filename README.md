# AI Voice Privacy Protection

**Real-time voice privacy system that protects voice biometrics, emotional patterns, and spoken PII while maintaining natural conversation flow.**

## 🚀 Core Features

- **Real-Time Voice Biometric Anonymization** - < 50ms end-to-end latency
- **Contextual Privacy Intelligence** - Meeting, Personal, Public, Emergency modes
- **Multi-Modal PII Detection** - SSN, credit cards, addresses, emotional markers
- **TikTok Live Integration** - Seamless streaming privacy protection
- **Cross-Platform Support** - iOS, Android, Web, Desktop
- **Edge AI Optimization** - On-device processing for maximum privacy

## 🎯 Latest Technologies Integration

- **Whisper-v3 + Custom Fine-tuning** - Ultra-fast speech recognition
- **StyleTTS2 + Voice Conversion** - Real-time voice anonymization
- **Pyannote 3.0** - Advanced speaker diarization
- **WavLM + Custom Heads** - Emotion detection and neutralization
- **ONNX Runtime Mobile** - Optimized on-device processing

## 🏗️ Project Structure

```
├── core/                    # Core AI processing pipeline
│   ├── audio/              # Audio processing and VAD
│   ├── models/             # AI models and inference
│   ├── privacy/            # Privacy protection algorithms
│   └── utils/              # Shared utilities
├── integrations/           # Platform integrations
│   ├── tiktok/            # TikTok Live integration
│   ├── mobile/            # iOS/Android apps
│   └── web/               # Web browser extension
├── ui/                     # Lynx-powered UI components
├── tests/                  # Test suites
└── docs/                   # Documentation
```

## 🔧 Quick Start

1. **Install Dependencies**
   ```bash
   npm install
   pip install -r requirements.txt
   ```

2. **Run Development Server**
   ```bash
   npm run dev
   ```

3. **Test Voice Processing**
   ```bash
   python tests/test_voice_shield.py
   ```

## 📱 TikTok Live Demo

Experience VoiceShield protection during live streaming with automatic:
- Background voice filtering
- PII detection and masking  
- Real-time privacy visualization
- Emergency privacy controls

## 🛠️ Development

This project uses cutting-edge AI technologies for real-time voice privacy protection. See `/docs` for detailed technical documentation.

Built for TikTok TechJam 2025 🎵
