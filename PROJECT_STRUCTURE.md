# TikTok TechJam 2025 - VoiceShield Project Structure

## 📁 Project Organization

```
C:\tiktok-techjam-2025\
├── 📁 src/                          # Source code (React + Core)
│   ├── 📁 core/                     # Core VoiceShield functionality
│   ├── 📁 server/                   # Express server & WebSocket
│   ├── 📁 ui/                       # React components (consolidated)
│   ├── 📁 utils/                    # Shared utilities
│   ├── index.js                     # Main module exports
│   └── index.jsx                    # React app entry point
│
├── 📁 config/                       # Configuration files
│   ├── package.json                 # Node.js dependencies
│   ├── requirements.txt             # Python dependencies
│   ├── vite.config.js              # Build configuration
│   └── webpack.config.js            # Bundler configuration
│
├── 📁 server/                       # Server components
│   ├── main.py                      # Primary Python server
│   ├── python_server.py             # Python backend
│   └── dev_server.js               # Development server
│
├── 📁 training/                     # ML Training scripts
│   ├── master_training.py           # Main training orchestrator
│   ├── advanced_audio_training.py   # Advanced audio training
│   ├── sota_fine_tuning.py         # SOTA fine-tuning
│   └── [11 other training scripts]
│
├── 📁 demos/                        # Demonstration scripts
│   ├── LIVE_VOICE_DEMO.py          # Live voice demo
│   ├── COMPREHENSIVE_DEMO.py       # Full feature demo
│   ├── PRODUCTION_DEMO.py          # Production demo
│   └── [4 other demo scripts]
│
├── 📁 tests/                        # Test files
│   ├── test_voice_shield.py         # Core tests
│   ├── test_imports.py             # Import tests
│   └── test_real_processing.py     # Processing tests
│
├── 📁 logs/                         # Training & application logs
├── 📁 docs/                         # Documentation & reports
├── 📁 models/                       # Trained models
├── 📁 assets/                       # Static assets
├── 📁 public/                       # Public web assets
├── 📁 scripts/                      # Utility scripts
├── 📁 wandb/                        # Weights & Biases logs
│
├── README.md                        # Main project documentation
├── LICENSE                          # Project license
└── node_modules/                    # Node.js dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Python dependencies
pip install -r config/requirements.txt

# Node.js dependencies
cd config && npm install
```

### 2. Run Demo
```bash
python demos/LIVE_VOICE_DEMO.py
```

### 3. Start Development Server
```bash
node server/dev_server.js
```

## 📝 Directory Descriptions

- **`src/`** - Main source code with modular architecture
- **`config/`** - All configuration files centralized
- **`server/`** - Server-side components (Python & Node.js)
- **`training/`** - ML training scripts and experiments
- **`demos/`** - Interactive demonstrations of features
- **`tests/`** - Automated tests and validation scripts
- **`logs/`** - Training logs and application logs
- **`docs/`** - Project documentation and reports

## 🔧 Development

Each directory contains its own README.md with specific instructions.
Check individual directory documentation for detailed usage.
