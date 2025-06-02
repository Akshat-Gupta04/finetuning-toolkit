# 🚀 Diffusion Model Finetuning Toolkit

A **standardized, normalized, and regularized pipeline** for finetuning FLUX, Wan2.1, and other diffusion models with LoRA/QLoRA/DoRA support.

## 🎯 **Features**

### **✅ Supported Models**
- **FLUX.1-dev** - Text-to-Image generation
- **Wan2.1 I2V** - Image-to-Video generation  
- **Wan2.1 T2V** - Text-to-Video generation

### **✅ Training Methods**
- **LoRA** - Low-Rank Adaptation (full precision)
- **QLoRA** - 4-bit quantized LoRA (50% memory reduction)
- **DoRA** - Weight-Decomposed LoRA (enhanced quality)

### **✅ Key Features**
- 🔧 **Standardized Pipeline** - Unified interface for all models
- 📊 **Auto-Captioning** - BLIP2/BLIP automatic caption generation
- 🎨 **Variable-Size Training** - Preserves aspect ratios, no cropping
- ⚡ **A40 Optimized** - Memory-efficient training for 48GB VRAM
- 📈 **Real-time Monitoring** - W&B, TensorBoard integration
- 🔄 **Resume Training** - Checkpoint support
- 🎛️ **CLI Interface** - Easy command-line usage

## 🏗️ **Architecture**

```
toolkit/
├── core.py              # Base classes and configurations
├── utils.py             # Utility functions and LoRA configs
├── pipeline.py          # Main training/inference pipelines
├── models/              # Model implementations
│   ├── flux_model.py
│   ├── wan21_i2v_model.py
│   └── wan21_t2v_model.py
├── datasets/            # Dataset implementations
│   ├── text_to_image_dataset.py
│   ├── image_to_video_dataset.py
│   └── text_to_video_dataset.py
└── trainers/            # Trainer implementations
    ├── flux_trainer.py
    ├── wan21_i2v_trainer.py
    └── wan21_t2v_trainer.py
```

## 🚀 **Quick Start**

### **1. Installation**

```bash
# Clone repository
git clone <repository-url>
cd diffusion-toolkit

# Install dependencies
pip install -r requirements.txt
```

### **2. Create Configuration**

```bash
# Create default FLUX config
python toolkit_cli.py create-config flux --output configs/my_flux_config.yaml

# Create Wan2.1 I2V config
python toolkit_cli.py create-config wan2_1_i2v --output configs/my_wan21_i2v_config.yaml

# Create Wan2.1 T2V config
python toolkit_cli.py create-config wan2_1_t2v --output configs/my_wan21_t2v_config.yaml
```

### **3. Prepare Dataset (Optional)**

```bash
# Auto-caption images for FLUX training
python toolkit_cli.py prepare-data \
    --data-dir ./my_images \
    --dataset-type text_to_image \
    --captioning-model blip2

# Auto-caption videos for Wan2.1 T2V training
python toolkit_cli.py prepare-data \
    --data-dir ./my_videos \
    --dataset-type text_to_video \
    --captioning-model blip2
```

### **4. Train Models**

#### **FLUX Text-to-Image**
```bash
# Standard LoRA training
python toolkit_cli.py train \
    --config configs/flux_config.yaml \
    --data-dir ./my_images \
    --output-dir ./outputs/flux_lora

# QLoRA training (memory efficient)
python toolkit_cli.py train \
    --config configs/flux_config.yaml \
    --data-dir ./my_images \
    --use-qlora \
    --lora-rank 32 \
    --output-dir ./outputs/flux_qlora

# DoRA training (enhanced quality)
python toolkit_cli.py train \
    --config configs/flux_config.yaml \
    --data-dir ./my_images \
    --use-dora \
    --lora-rank 64 \
    --output-dir ./outputs/flux_dora
```

#### **Wan2.1 Image-to-Video**
```bash
python toolkit_cli.py train \
    --config configs/wan21_i2v_config.yaml \
    --data-dir ./my_image_video_pairs \
    --output-dir ./outputs/wan21_i2v_lora
```

#### **Wan2.1 Text-to-Video**
```bash
python toolkit_cli.py train \
    --config configs/wan21_t2v_config.yaml \
    --data-dir ./my_videos \
    --output-dir ./outputs/wan21_t2v_lora
```

### **5. Run Inference**

```bash
# FLUX inference
python toolkit_cli.py inference flux \
    --model-path ./outputs/flux_lora/final_model \
    --prompt "A beautiful landscape with mountains and lakes" \
    --output generated_image.png \
    --height 1024 \
    --width 1024

# Wan2.1 I2V inference (coming soon)
python toolkit_cli.py inference wan2_1_i2v \
    --model-path ./outputs/wan21_i2v_lora/final_model \
    --image input_image.jpg \
    --prompt "The person starts walking" \
    --output generated_video.mp4

# Wan2.1 T2V inference (coming soon)
python toolkit_cli.py inference wan2_1_t2v \
    --model-path ./outputs/wan21_t2v_lora/final_model \
    --prompt "A serene lake with gentle ripples" \
    --output generated_video.mp4
```

## ⚙️ **Configuration**

### **Model Configuration**
```yaml
model:
  model_name: "flux"  # flux, wan2_1_i2v, wan2_1_t2v
  pretrained_model_path: "black-forest-labs/FLUX.1-dev"
  
  # LoRA settings
  use_lora: true
  lora_type: "lora"  # lora, qlora, dora
  lora_rank: 64
  lora_alpha: 64
  lora_dropout: 0.1
  
  # Quantization (for QLoRA)
  use_4bit: false
  bnb_4bit_compute_dtype: "bfloat16"
```

### **Dataset Configuration**
```yaml
dataset:
  dataset_type: "text_to_image"  # text_to_image, image_to_video, text_to_video
  data_dir: "./data"
  resolution: 1024
  
  # Auto-captioning
  auto_caption: true
  captioning_model: "blip2"
  quality_threshold: 0.7
  
  # Variable size training
  variable_size: true
```

### **Training Configuration**
```yaml
training:
  output_dir: "./outputs"
  max_train_steps: 1000
  train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  mixed_precision: "bf16"
  
  # Monitoring
  logging_steps: 10
  validation_steps: 500
  checkpointing_steps: 500
```

## 📊 **Memory Usage (A40 GPU)**

| Model | Resolution | LoRA Type | Batch Size | Memory Usage |
|-------|------------|-----------|------------|--------------|
| FLUX | 1024x1024 | LoRA | 1 | ~35GB |
| FLUX | 1024x1024 | QLoRA | 2 | ~25GB |
| FLUX | 1024x1024 | DoRA | 1 | ~37GB |
| Wan2.1 I2V | 720p x16f | LoRA | 1 | ~40GB |
| Wan2.1 I2V | 720p x16f | QLoRA | 2 | ~28GB |
| Wan2.1 T2V | 720p x16f | LoRA | 1 | ~42GB |
| Wan2.1 T2V | 720p x16f | QLoRA | 2 | ~30GB |

## 🔧 **Advanced Usage**

### **Python API**
```python
from toolkit.pipeline import TrainingPipeline
from toolkit.core import ModelConfig, DatasetConfig, TrainingConfig

# Create configurations
model_config = ModelConfig(
    model_name="flux",
    pretrained_model_path="black-forest-labs/FLUX.1-dev",
    use_lora=True,
    lora_type="qlora",
    lora_rank=32
)

dataset_config = DatasetConfig(
    dataset_type="text_to_image",
    data_dir="./data",
    resolution=1024,
    auto_caption=True
)

training_config = TrainingConfig(
    output_dir="./outputs",
    max_train_steps=1000,
    train_batch_size=1,
    learning_rate=1e-4
)

# Create and run pipeline
config = {
    "model": model_config.__dict__,
    "dataset": dataset_config.__dict__,
    "training": training_config.__dict__
}

pipeline = TrainingPipeline(config)
pipeline.run()
```

### **Custom Dataset**
```python
from toolkit.core import BaseDataset, DatasetConfig

class CustomDataset(BaseDataset):
    def prepare_data(self):
        # Load your custom data
        pass
    
    def __getitem__(self, idx):
        # Return item
        return {
            'pixel_values': image_tensor,
            'caption': caption_text
        }
    
    def collate_fn(self, batch):
        # Custom collation
        pass
```

## 📈 **Monitoring & Logging**

### **W&B Integration**
```yaml
training:
  report_to: "wandb"

wandb:
  project_name: "my-diffusion-project"
  run_name: "flux-lora-experiment"
  tags: ["flux", "lora", "text2image"]
```

### **TensorBoard Integration**
```yaml
training:
  report_to: "tensorboard"
  logging_dir: "./logs"
```

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **CUDA OOM Error**
   ```bash
   # Use QLoRA for memory efficiency
   python toolkit_cli.py train --config config.yaml --use-qlora --lora-rank 32
   ```

2. **Slow Training**
   ```bash
   # Increase batch size with gradient accumulation
   python toolkit_cli.py train --config config.yaml --batch-size 2 --gradient-accumulation-steps 4
   ```

3. **Poor Quality Results**
   ```bash
   # Use DoRA for better quality
   python toolkit_cli.py train --config config.yaml --use-dora --lora-rank 64
   ```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License.

## 🙏 **Acknowledgments**

- **FLUX.1** by Black Forest Labs
- **Wan2.1** by Wan-AI
- **LoRA** by Microsoft Research
- **QLoRA** by University of Washington
- **DoRA** by NVIDIA Research

---

**🎉 Happy Finetuning!** 🚀
