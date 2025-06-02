#!/usr/bin/env python3
"""
Simple FLUX QLoRA training script
Optimized for your setup with images in /workspace/data/images
"""

import os
import sys
import subprocess
from pathlib import Path


def train_flux_qlora():
    """Train FLUX with QLoRA on your images"""

    print("🚀 FLUX QLoRA Training Setup")
    print("=" * 50)

    # Configuration
    config = {
        "model": "flux",
        "input_dir": "/workspace/data/images",
        "output_dir": "/workspace/outputs/flux_qlora",
        "max_steps": 1000,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "lora_rank": 32,
        "lora_alpha": 32,
        "checkpointing_steps": 250,  # Save every 250 steps
        "validation_steps": 250,     # Test every 250 steps
        "logging_steps": 10
    }

    # Verify images exist
    image_dir = Path(config["input_dir"])
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return

    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(list(image_dir.glob(f"*{ext}")))
        images.extend(list(image_dir.glob(f"*{ext.upper()}")))

    print(f"📁 Image directory: {image_dir}")
    print(f"🖼️  Found {len(images)} images")

    if len(images) == 0:
        print("❌ No images found! Please add images to /workspace/data/images/")
        return

    # Show first few images
    print(f"📋 Sample images:")
    for i, img in enumerate(images[:5]):
        print(f"  {i+1}. {img.name}")
    if len(images) > 5:
        print(f"  ... and {len(images) - 5} more")

    # Create output directory
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Build training command with correct arguments
    cmd = [
        "python", "train.py",
        "--model", config["model"],
        "--input_dir", config["input_dir"],
        "--prepare_dataset",  # Auto-generate captions
        "--config", "config/flux_qlora_config.yaml",  # Use QLoRA config
        "--data_dir", "/workspace/finetuning-toolkit/data/train",
        "--output_dir", config["output_dir"],
        "--batch_size", str(config["batch_size"]),
        "--mixed_precision", "bf16"
    ]

    print(f"\n🎯 Training Configuration:")
    print(f"  Model: FLUX with QLoRA")
    print(f"  Images: {len(images)} files")
    print(f"  LoRA Rank: {config['lora_rank']}")
    print(f"  Max Steps: {config['max_steps']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Checkpoints: Every {config['checkpointing_steps']} steps")
    print(f"  Output: {config['output_dir']}")

    print(f"\n🚀 Starting training...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    # Start training
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Stream output
        for line in process.stdout:
            print(line.rstrip())

        process.wait()

        if process.returncode == 0:
            print(f"\n✅ Training completed successfully!")
            print(f"📁 Model saved to: {config['output_dir']}")
        else:
            print(f"\n❌ Training failed with return code: {process.returncode}")

    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        process.terminate()
    except Exception as e:
        print(f"\n❌ Training error: {e}")


def main():
    """Main function"""

    # Check if we're in the right directory
    if not os.path.exists("train.py"):
        print("❌ train.py not found! Please run from the project root directory.")
        return

    # Check if .env exists
    if not os.path.exists(".env"):
        print("❌ .env file not found! Please run 'python fix_env.py' first.")
        return

    # Start training
    train_flux_qlora()


if __name__ == "__main__":
    main()
