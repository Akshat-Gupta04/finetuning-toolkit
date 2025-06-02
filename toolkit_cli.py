#!/usr/bin/env python3
"""
Diffusion Model Finetuning Toolkit - Command Line Interface
A standardized, normalized pipeline for finetuning FLUX, Wan2.1, and other diffusion models
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import yaml

# Add toolkit to path
sys.path.insert(0, str(Path(__file__).parent))

from toolkit.pipeline import TrainingPipeline, InferencePipeline, DataPreparationPipeline
from toolkit.core import ModelConfig, DatasetConfig, TrainingConfig
from toolkit.utils import setup_logging, validate_config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def create_default_config(model_type: str, output_path: str) -> Dict[str, Any]:
    """Create default configuration for a model type"""
    
    # Base configuration
    config = {
        "model": {
            "model_name": model_type,
            "pretrained_model_path": "",
            "cache_dir": "./cache",
            "torch_dtype": "bfloat16",
            
            # LoRA configuration
            "use_lora": True,
            "lora_type": "lora",  # lora, qlora, dora
            "lora_rank": 64,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            
            # Quantization
            "use_4bit": False,
            "use_8bit": False,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True
        },
        
        "dataset": {
            "dataset_type": "text_to_image",
            "data_dir": "./data",
            "image_column": "image",
            "caption_column": "caption",
            "resolution": 512,
            "center_crop": True,
            "random_flip": 0.0,
            "max_sequence_length": 512,
            
            # Auto-captioning
            "auto_caption": False,
            "captioning_model": "blip2",
            "quality_threshold": 0.7,
            
            # Variable size training
            "variable_size": True
        },
        
        "training": {
            "output_dir": "./outputs",
            "logging_dir": "./logs",
            
            # Training parameters
            "num_train_epochs": 1,
            "max_train_steps": 1000,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 100,
            
            # Optimization
            "use_8bit_adam": False,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 0.01,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            
            # Mixed precision
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            
            # Logging and validation
            "logging_steps": 10,
            "validation_steps": 500,
            "checkpointing_steps": 500,
            
            # Reproducibility
            "seed": 42,
            
            # Hardware
            "dataloader_num_workers": 4,
            "pin_memory": True
        }
    }
    
    # Model-specific configurations
    if model_type == "flux":
        config["model"]["pretrained_model_path"] = "black-forest-labs/FLUX.1-dev"
        config["dataset"]["dataset_type"] = "text_to_image"
        config["dataset"]["resolution"] = 1024
        config["training"]["train_batch_size"] = 1
        config["training"]["gradient_accumulation_steps"] = 8
        
    elif model_type == "wan2_1_i2v":
        config["model"]["pretrained_model_path"] = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
        config["dataset"]["dataset_type"] = "image_to_video"
        config["dataset"]["resolution"] = (1280, 720)
        config["dataset"]["num_frames"] = 16
        config["training"]["train_batch_size"] = 1
        config["training"]["gradient_accumulation_steps"] = 4
        
    elif model_type == "wan2_1_t2v":
        config["model"]["pretrained_model_path"] = "Wan-AI/Wan2.1-T2V-14B-720P-Diffusers"
        config["dataset"]["dataset_type"] = "text_to_video"
        config["dataset"]["resolution"] = (1280, 720)
        config["dataset"]["num_frames"] = 16
        config["training"]["train_batch_size"] = 1
        config["training"]["gradient_accumulation_steps"] = 4
    
    return config


def cmd_train(args):
    """Training command"""
    setup_logging(log_level="INFO")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.data_dir:
        config["dataset"]["data_dir"] = args.data_dir
    if args.batch_size:
        config["training"]["train_batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.max_steps:
        config["training"]["max_train_steps"] = args.max_steps
    if args.lora_rank:
        config["model"]["lora_rank"] = args.lora_rank
    if args.lora_alpha:
        config["model"]["lora_alpha"] = args.lora_alpha
    if args.use_qlora:
        config["model"]["lora_type"] = "qlora"
        config["model"]["use_4bit"] = True
    if args.use_dora:
        config["model"]["lora_type"] = "dora"
    
    # Create and run training pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run(resume_from_checkpoint=args.resume_from_checkpoint)


def cmd_inference(args):
    """Inference command"""
    setup_logging(log_level="INFO")
    
    # Create model config
    model_config = ModelConfig(
        model_name=args.model_type,
        pretrained_model_path=args.model_path,
        use_lora=args.lora_path is not None
    )
    
    # Create inference pipeline
    pipeline = InferencePipeline(model_config, args.model_path)
    pipeline.setup()
    
    # Generate based on model type
    if args.model_type == "flux":
        if not args.prompt:
            raise ValueError("Prompt required for FLUX inference")
        
        result = pipeline.generate(
            prompt=args.prompt,
            height=args.height or 1024,
            width=args.width or 1024,
            guidance_scale=args.guidance_scale or 7.5,
            num_inference_steps=args.num_steps or 20
        )
        
        # Save result
        output_path = args.output or "generated_image.png"
        result.save(output_path)
        print(f"Generated image saved to: {output_path}")
        
    else:
        print(f"Inference for {args.model_type} not yet implemented")


def cmd_prepare_data(args):
    """Data preparation command"""
    setup_logging(log_level="INFO")
    
    # Create basic config for data preparation
    config = {
        "dataset": {
            "data_dir": args.data_dir,
            "auto_caption": True,
            "captioning_model": args.captioning_model or "blip2",
            "quality_threshold": args.quality_threshold or 0.7,
            "dataset_type": args.dataset_type or "text_to_image"
        }
    }
    
    # Create and run data preparation pipeline
    pipeline = DataPreparationPipeline(config)
    pipeline.prepare_dataset()
    
    # Validate dataset
    validation_results = pipeline.validate_dataset()
    print("Dataset validation results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")


def cmd_create_config(args):
    """Create configuration command"""
    config = create_default_config(args.model_type, args.output)
    save_config(config, args.output)
    print(f"Default configuration for {args.model_type} saved to: {args.output}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Diffusion Model Finetuning Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default config
  python toolkit_cli.py create-config flux --output flux_config.yaml
  
  # Prepare dataset with auto-captioning
  python toolkit_cli.py prepare-data --data-dir ./images --dataset-type text_to_image
  
  # Train FLUX model
  python toolkit_cli.py train --config flux_config.yaml --data-dir ./data
  
  # Train with QLoRA
  python toolkit_cli.py train --config flux_config.yaml --use-qlora --lora-rank 32
  
  # Run inference
  python toolkit_cli.py inference flux --model-path ./outputs/final_model --prompt "A beautiful landscape"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train a diffusion model")
    train_parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    train_parser.add_argument("--output-dir", type=str, help="Output directory")
    train_parser.add_argument("--data-dir", type=str, help="Data directory")
    train_parser.add_argument("--batch-size", type=int, help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    train_parser.add_argument("--lora-rank", type=int, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    train_parser.add_argument("--use-qlora", action="store_true", help="Use QLoRA (4-bit)")
    train_parser.add_argument("--use-dora", action="store_true", help="Use DoRA")
    train_parser.add_argument("--resume-from-checkpoint", type=str, help="Resume from checkpoint")
    train_parser.set_defaults(func=cmd_train)
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument("model_type", choices=["flux", "wan2_1_i2v", "wan2_1_t2v"], help="Model type")
    inference_parser.add_argument("--model-path", type=str, required=True, help="Model path")
    inference_parser.add_argument("--lora-path", type=str, help="LoRA weights path")
    inference_parser.add_argument("--prompt", type=str, help="Text prompt")
    inference_parser.add_argument("--image", type=str, help="Input image (for I2V)")
    inference_parser.add_argument("--output", type=str, help="Output path")
    inference_parser.add_argument("--height", type=int, help="Output height")
    inference_parser.add_argument("--width", type=int, help="Output width")
    inference_parser.add_argument("--guidance-scale", type=float, help="Guidance scale")
    inference_parser.add_argument("--num-steps", type=int, help="Number of inference steps")
    inference_parser.set_defaults(func=cmd_inference)
    
    # Data preparation command
    data_parser = subparsers.add_parser("prepare-data", help="Prepare dataset with auto-captioning")
    data_parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    data_parser.add_argument("--dataset-type", type=str, choices=["text_to_image", "image_to_video", "text_to_video"], help="Dataset type")
    data_parser.add_argument("--captioning-model", type=str, choices=["blip", "blip2"], help="Captioning model")
    data_parser.add_argument("--quality-threshold", type=float, help="Quality threshold")
    data_parser.set_defaults(func=cmd_prepare_data)
    
    # Config creation command
    config_parser = subparsers.add_parser("create-config", help="Create default configuration")
    config_parser.add_argument("model_type", choices=["flux", "wan2_1_i2v", "wan2_1_t2v"], help="Model type")
    config_parser.add_argument("--output", type=str, required=True, help="Output config file path")
    config_parser.set_defaults(func=cmd_create_config)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run command
    try:
        args.func(args)
    except Exception as e:
        logging.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
