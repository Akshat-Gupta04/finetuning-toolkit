# 🎨 Diffusion Training Studio

A clean, modern web interface for training FLUX and Wan2.1 diffusion models with automatic dataset preparation and real-time monitoring.

## ✨ Features

- **🎯 Web Interface**: Beautiful, animated Flask web UI
- **🤖 Auto Captioning**: AI-powered image captioning (BLIP/BLIP2)
- **📏 Variable Size Training**: No cropping, preserves aspect ratios
- **⚡ Real-time Monitoring**: Live training progress and logs
- **💾 Easy Downloads**: One-click model downloads
- **🔧 A40 Optimized**: Configured for cloud A40 GPUs

## 🚀 Quick Start

### 1. Installation
```bash
git clone <repository>
cd finetune
pip install -r requirements.txt
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env with your HF_TOKEN
```

### 3. Start Web Interface
```bash
python app.py
```

### 4. Open Browser
Navigate to `http://localhost:5000`

## 🎯 Usage

### Web Interface
1. **Upload Images**: Drag & drop or browse images
2. **Select Model**: Choose FLUX or Wan2.1
3. **Configure Training**: Set parameters (batch size, learning rate, etc.)
4. **Start Training**: Click "Start Training"
5. **Monitor Progress**: Real-time logs and progress bars
6. **Download Model**: Get trained model when complete

### Command Line (Alternative)
```bash
# FLUX training with auto dataset prep
python train.py --model flux --input_dir /path/to/images --prepare_dataset

# Wan2.1 training
python train.py --model wan2.1 --data_dir ./data/train

# Generate images/videos
python inference.py --model flux --prompt "beautiful landscape" --lora_path ./outputs/lora
python inference.py --model wan2.1 --image_path input.jpg --prompt "flowing water" --lora_path ./outputs/lora
```

## 📁 Clean File Structure

```
finetune/
├── 🌐 Web Interface
│   ├── app.py                  # Flask web application
│   └── templates/              # HTML templates
│       ├── base.html           # Base template with animations
│       ├── index.html          # Dashboard
│       ├── train.html          # Training interface
│       └── monitor.html        # Real-time monitoring
│
├── 🚀 Core Training
│   ├── train.py                # Main training entry point
│   ├── inference.py            # Unified inference
│   └── src/
│       ├── dataset_auto.py     # Auto dataset preparation
│       ├── train_unified.py    # Unified training logic
│       └── config_manager.py   # Configuration management
│
├── ⚙️ Configuration
│   ├── config/
│   │   └── unified_config.yaml # A40-optimized settings
│   ├── requirements.txt        # All dependencies
│   └── .env.example           # Environment template
│
└── 📊 Data & Models
    ├── data/                   # Training datasets
    ├── outputs/                # Trained models
    └── uploads/                # Web uploads
```

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# Required
HF_TOKEN=your_huggingface_token

# Optional
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=your_project_name
OUTPUT_DIR=./outputs
CACHE_DIR=./cache
```

### Model Support
- **FLUX**: Text-to-image generation (512px-2048px) with LoRA/QLoRA/DoRA
- **Wan2.1**: Image-to-video generation (720p-1080p) with LoRA/QLoRA/DoRA
- **Text2Video**: Text-to-video generation (256px-512px, 16 frames) with LoRA/QLoRA/DoRA

### LoRA/QLoRA/DoRA Features
- **LoRA**: Standard Low-Rank Adaptation for efficient training
- **QLoRA**: 4-bit quantized LoRA for maximum memory efficiency (uses ~50% less VRAM)
- **DoRA**: Weight-Decomposed LoRA for enhanced quality
- **Configurable Ranks**: 16, 32, 64, 128, 256 rank options
- **Model-Specific Targets**: Optimized target modules per model type
- **Web Interface**: Easy LoRA type selection and parameter tuning

## 🔧 A40 Optimizations

### Recommended Settings
| Model | Resolution | LoRA Type | Batch Size | Memory Usage | Training Time |
|-------|------------|-----------|------------|--------------|---------------|
| FLUX | 1024x1024 | LoRA | 4 | ~35GB | 2-4 hours |
| FLUX | 1024x1024 | QLoRA | 8 | ~18GB | 2-4 hours |
| FLUX | 512x512 | LoRA | 8 | ~20GB | 1-2 hours |
| FLUX | 512x512 | QLoRA | 16 | ~10GB | 1-2 hours |
| Wan2.1 | 720p | LoRA | 2 | ~40GB | 4-6 hours |
| Wan2.1 | 720p | QLoRA | 4 | ~20GB | 4-6 hours |
| Text2Video | 512px | LoRA | 4 | ~30GB | 3-5 hours |
| Text2Video | 512px | QLoRA | 8 | ~15GB | 3-5 hours |

### Memory Features
- **BF16 Precision**: Faster training on A40
- **Gradient Checkpointing**: Reduces memory usage
- **XFormers**: Memory efficient attention
- **Large Batch Support**: Utilizes 48GB VRAM

## 🎨 Web Interface Features

### 📊 Dashboard
- Model overview and statistics
- Recent training sessions
- Quick start buttons
- Animated UI elements

### 🚀 Training Page
- Drag & drop file upload
- Model selection (FLUX/Wan2.1/Text2Video)
- LoRA/QLoRA/DoRA configuration
- Parameter tuning (rank, alpha, dropout)
- Real-time progress monitoring

### 📈 Monitor Page
- All training sessions
- Live progress updates
- Detailed logs and metrics
- Model downloads

## 🛠️ API Endpoints

- `POST /api/upload` - Upload training images
- `POST /api/train` - Start training session
- `GET /api/status/<session_id>` - Get training status
- `GET /api/logs/<session_id>` - Get training logs
- `GET /api/download/<session_id>` - Download trained model
- `GET /api/sessions` - List all sessions

## 📊 Real-time Features

- **WebSocket Updates**: Live training progress
- **Auto Refresh**: Session list updates every 30s
- **Progress Bars**: Visual training progress
- **Log Streaming**: Real-time training logs
- **Notifications**: Training completion alerts

## 🎯 Training Tips

### Dataset Preparation
- Use 50-200 high-quality images
- Mixed aspect ratios work well
- Higher quality threshold = better results
- BLIP2 provides better captions but slower

### Training Parameters
- Batch size 4 recommended for A40
- Learning rate 1e-4 works well
- 5000 steps for good results
- LoRA rank 64 balances quality/speed

### Performance
- Variable-size training preserves quality
- BF16 precision recommended for A40
- Enable gradient checkpointing for memory
- Use XFormers for efficiency

## 🔍 Troubleshooting

### Common Issues
- **CUDA OOM**: Reduce batch size
- **Slow upload**: Check file sizes
- **Training stuck**: Check logs for errors
- **Model access**: Verify HF_TOKEN

### Memory Optimization
- Use batch size 1-2 for limited memory
- Enable gradient checkpointing
- Use QLoRA for extreme efficiency
- Monitor GPU usage in real-time

## 🎉 Success Examples

### Training Times (A40)
- **FLUX 1024px**: 2-4 hours (100 images)
- **FLUX 512px**: 1-2 hours (100 images)
- **Wan2.1 720p**: 4-6 hours (50 videos)

### Use Cases
- Art style transfer
- Product photography
- Character design
- Brand assets
- Video animation

## 🌟 Key Advantages

### Minimal Codebase
- **5 Core Files**: Easy to maintain and understand
- **Clean Structure**: No scattered demos or utilities
- **Single Interface**: Web UI handles everything
- **Unified Logic**: One system for both models

### Production Ready
- **A40 Optimized**: Maximum cloud GPU utilization
- **Real-time Monitoring**: Live progress and logs
- **Error Handling**: Graceful failure recovery
- **Auto Downloads**: Easy model retrieval

### Developer Friendly
- **Modern UI**: Beautiful, responsive interface
- **WebSocket Updates**: Real-time communication
- **API Endpoints**: Programmatic access
- **Clean Code**: Well-structured and documented

---

**Ready to train? Start the web interface:**
```bash
python app.py
```

Then open `http://localhost:5000` and start training! 🚀

**From raw images to trained models in a beautiful web interface - optimized for A40 GPUs!** ✨
