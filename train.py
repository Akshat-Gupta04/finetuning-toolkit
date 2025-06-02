#!/usr/bin/env python3
"""
Unified entry point for diffusion model training
Handles both Wan2.1 and FLUX with automatic dataset preparation
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from dataset_auto import prepare_dataset
    from train_unified import train_model
    from config_manager import load_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="Unified diffusion model training")

    # Model selection
    parser.add_argument("--model", type=str, required=True, choices=["flux", "wan2_1_i2v", "wan2_1_t2v"],
                       help="Model type to train")

    # Dataset preparation
    parser.add_argument("--input_dir", type=str, help="Input directory with raw images")
    parser.add_argument("--prepare_dataset", action="store_true",
                       help="Automatically prepare dataset from input_dir")
    parser.add_argument("--captioning_model", type=str, default="blip2",
                       choices=["blip2", "blip"], help="Captioning model for dataset preparation")

    # Training configuration
    parser.add_argument("--config", type=str, default="config/unified_config.yaml",
                       help="Training configuration file")
    parser.add_argument("--data_dir", type=str, default="./data/train",
                       help="Training data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for trained model")

    # Training options
    parser.add_argument("--variable_size", action="store_true", default=True,
                       help="Enable variable-size training (default: True)")
    parser.add_argument("--fixed_size", action="store_true",
                       help="Use fixed-size training instead of variable-size")
    parser.add_argument("--resume_from_checkpoint", type=str,
                       help="Resume training from checkpoint")

    # Hardware optimization
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Override gradient accumulation")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"],
                       help="Mixed precision mode")

    # Quality settings
    parser.add_argument("--min_resolution", type=int, default=512,
                       help="Minimum image resolution for dataset preparation")
    parser.add_argument("--max_resolution", type=int, default=2048,
                       help="Maximum image resolution for dataset preparation")
    parser.add_argument("--quality_threshold", type=float, default=0.7,
                       help="Quality threshold for dataset preparation")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Validate arguments
    if args.prepare_dataset and not args.input_dir:
        parser.error("--input_dir is required when --prepare_dataset is used")

    # Handle variable size vs fixed size
    variable_size = args.variable_size and not args.fixed_size

    logger.info(f"🚀 Starting {args.model.upper()} training")
    logger.info(f"📊 Variable-size training: {'enabled' if variable_size else 'disabled'}")

    # Step 1: Prepare dataset if requested
    if args.prepare_dataset:
        logger.info(f"📁 Preparing dataset from {args.input_dir}")

        stats = prepare_dataset(
            input_dir=args.input_dir,
            output_dir=args.data_dir,
            model_type=args.model,
            captioning_model=args.captioning_model,
            min_resolution=args.min_resolution,
            max_resolution=args.max_resolution,
            quality_threshold=args.quality_threshold,
            batch_size=8  # Optimized for A40
        )

        logger.info(f"✅ Dataset prepared: {stats['processed']} images, {stats['captions_generated']} captions")

    # Step 2: Load and modify config
    logger.info(f"📋 Loading config from {args.config}")
    config = load_config(args.config)

    # Apply model-specific overrides
    if "model_overrides" in config and args.model in config["model_overrides"]:
        model_overrides = config["model_overrides"][args.model]
        config["model"].update(model_overrides)
        logger.info(f"Applied {args.model} model overrides")

    # Apply command line overrides
    config["dataset"]["train_data_dir"] = args.data_dir
    config["training"]["output_dir"] = args.output_dir

    if args.batch_size:
        config["training"]["train_batch_size"] = args.batch_size

    if args.gradient_accumulation_steps:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    if args.mixed_precision:
        config["training"]["mixed_precision"] = args.mixed_precision

    # A40 optimizations
    if "hardware_optimizations" in config:
        hw_opts = config["hardware_optimizations"]
        if hw_opts.get("enable_large_batch_training", False):
            # Optimize batch size for A40 based on model and resolution
            recommended_sizes = hw_opts.get("recommended_batch_sizes", {})

            if args.model == "flux":
                if variable_size or config["dataset"].get("resolution", 1024) >= 1024:
                    config["training"]["train_batch_size"] = recommended_sizes.get("flux_1024", 4)
                else:
                    config["training"]["train_batch_size"] = recommended_sizes.get("flux_512", 8)
            elif args.model == "wan2.1":
                if config["training"].get("max_height", 720) >= 1080:
                    config["training"]["train_batch_size"] = recommended_sizes.get("wan2_1_1080p", 1)
                else:
                    config["training"]["train_batch_size"] = recommended_sizes.get("wan2_1_720p", 2)

        logger.info(f"🔧 A40 optimized batch size: {config['training']['train_batch_size']}")

    # Step 3: Start training
    logger.info(f"🎯 Starting {args.model} training with variable_size={variable_size}")

    final_step = train_model(
        config=config,
        model_type=args.model,
        variable_size=variable_size,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    logger.info(f"🎉 Training completed at step {final_step}")
    logger.info(f"💾 Model saved to: {config['training']['output_dir']}")

    # Show next steps
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)
    print(f"Model: {args.model.upper()}")
    print(f"Output: {config['training']['output_dir']}")
    print(f"Variable-size: {'Yes' if variable_size else 'No'}")
    print(f"Final step: {final_step}")
    print()
    print("🚀 Next steps:")
    if args.model == "flux":
        print(f"1. Generate images:")
        print(f"   python inference.py --model flux --prompt 'your prompt' --lora_path {config['training']['output_dir']}/lora")
    elif args.model == "wan2.1":
        print(f"1. Generate videos:")
        print(f"   python inference.py --model wan2.1 --image_path input.jpg --prompt 'your prompt' --lora_path {config['training']['output_dir']}/lora")
    print()
    print("2. Monitor training logs:")
    print(f"   tail -f {config['training']['logging_dir']}/*.log")


if __name__ == "__main__":
    main()
