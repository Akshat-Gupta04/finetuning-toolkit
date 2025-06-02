#!/usr/bin/env python3
"""
Run FLUX QLoRA training with correct arguments
"""

import os
import subprocess
from pathlib import Path


def main():
    """Run FLUX QLoRA training"""
    
    print("🚀 FLUX QLoRA Training")
    print("=" * 50)
    
    # Check images
    image_dir = Path("data/images")
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(list(image_dir.glob(f"*{ext}")))
        images.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    print(f"📁 Found {len(images)} images in {image_dir}")
    
    if len(images) == 0:
        print("❌ No images found!")
        return
    
    # Create directories
    Path("/workspace/finetuning-toolkit/data/train").mkdir(parents=True, exist_ok=True)
    Path("/workspace/finetuning-toolkit/outputs/flux_qlora").mkdir(parents=True, exist_ok=True)
    
    # The exact command that works with train.py
    cmd = [
        "python", "train.py",
        "--model", "flux",
        "--input_dir", "data/images",
        "--prepare_dataset",
        "--config", "config/flux_qlora_config.yaml",
        "--data_dir", "/workspace/finetuning-toolkit/data/train",
        "--output_dir", "/workspace/finetuning-toolkit/outputs/flux_qlora",
        "--batch_size", "1",
        "--mixed_precision", "bf16"
    ]
    
    print(f"\n🎯 Configuration:")
    print(f"  Images: {len(images)} files")
    print(f"  Config: config/flux_qlora_config.yaml (QLoRA enabled)")
    print(f"  Output: /workspace/finetuning-toolkit/outputs/flux_qlora")
    print(f"  Auto-captioning: Enabled")
    
    print(f"\n🚀 Starting training...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Training completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted")


if __name__ == "__main__":
    main()
