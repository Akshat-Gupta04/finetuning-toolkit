#!/usr/bin/env python3
"""
Unified inference script for both Wan2.1 and FLUX models
Supports LoRA models and variable-size generation
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from config_manager import get_config_manager
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def load_flux_pipeline(model_path: str, lora_path: Optional[str] = None, device: str = "cuda"):
    """Load FLUX pipeline with optional LoRA"""
    from diffusers import FluxPipeline

    # Load base pipeline
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    )

    if device != "cuda":
        pipe = pipe.to(device)

    # Load LoRA if specified
    if lora_path and Path(lora_path).exists():
        pipe.load_lora_weights(lora_path)
        print(f"✅ Loaded LoRA weights from {lora_path}")

    return pipe


def load_wan2_1_pipeline(model_path: str, lora_path: Optional[str] = None, device: str = "cuda"):
    """Load Wan2.1 pipeline with optional LoRA"""
    from diffusers import I2VGenXLPipeline

    # Load base pipeline
    pipe = I2VGenXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    )

    if device != "cuda":
        pipe = pipe.to(device)

    # Load LoRA if specified
    if lora_path and Path(lora_path).exists():
        pipe.load_lora_weights(lora_path)
        print(f"✅ Loaded LoRA weights from {lora_path}")

    return pipe


def generate_flux_images(
    pipe,
    prompts: List[str],
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    num_images_per_prompt: int = 1,
    seed: Optional[int] = None
) -> List[Image.Image]:
    """Generate images with FLUX"""

    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    images = []

    for prompt in prompts:
        print(f"🎨 Generating: {prompt}")

        batch_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator
        ).images

        images.extend(batch_images)

    return images


def generate_wan2_1_videos(
    pipe,
    image_paths: List[str],
    prompts: List[str],
    negative_prompt: str = "",
    num_frames: int = 81,
    width: int = 1280,
    height: int = 720,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """Generate videos with Wan2.1"""

    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    videos = []

    for image_path, prompt in zip(image_paths, prompts):
        print(f"🎬 Generating video: {prompt}")

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((width, height))

        video_frames = pipe(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).frames[0]

        videos.append(video_frames)

    return videos


def save_images(images: List[Image.Image], output_dir: str, prefix: str = "generated"):
    """Save generated images"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, image in enumerate(images):
        filename = f"{prefix}_{i:04d}.png"
        filepath = output_path / filename
        image.save(filepath)
        saved_paths.append(str(filepath))

    return saved_paths


def save_videos(videos: List[np.ndarray], output_dir: str, prefix: str = "generated", fps: int = 16):
    """Save generated videos"""
    import imageio

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, video in enumerate(videos):
        filename = f"{prefix}_{i:04d}.mp4"
        filepath = output_path / filename

        # Convert to uint8 if needed
        if video.dtype != np.uint8:
            video = (video * 255).astype(np.uint8)

        imageio.mimsave(str(filepath), video, fps=fps)
        saved_paths.append(str(filepath))

    return saved_paths


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Unified inference for diffusion models")

    # Model selection
    parser.add_argument("--model", type=str, required=True, choices=["flux", "wan2_1_i2v", "wan2_1_t2v"],
                       help="Model type for inference")

    # Model paths
    parser.add_argument("--model_path", type=str, help="Base model path (default: use HF model)")
    parser.add_argument("--lora_path", type=str, help="LoRA weights path")

    # Generation parameters
    parser.add_argument("--prompt", type=str, action="append", required=True,
                       help="Generation prompt (can be used multiple times)")
    parser.add_argument("--negative_prompt", type=str, default="",
                       help="Negative prompt")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, help="Guidance scale")

    # FLUX-specific parameters
    parser.add_argument("--width", type=int, default=1024, help="Image width for FLUX")
    parser.add_argument("--height", type=int, default=1024, help="Image height for FLUX")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                       help="Number of images per prompt for FLUX")

    # Wan2.1-specific parameters
    parser.add_argument("--image_path", type=str, action="append",
                       help="Input image path for Wan2.1 (can be used multiple times)")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of video frames for Wan2.1")
    parser.add_argument("--fps", type=int, default=16, help="Video FPS for Wan2.1")

    # Output settings
    parser.add_argument("--output_dir", type=str, default="./outputs/inference",
                       help="Output directory")
    parser.add_argument("--output_prefix", type=str, default="generated",
                       help="Output filename prefix")

    # Hardware settings
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Setup config manager
    config_manager = get_config_manager()
    config_manager.setup_all()

    # Validate arguments
    if args.model == "wan2.1":
        if not args.image_path:
            parser.error("--image_path is required for Wan2.1")
        if len(args.image_path) != len(args.prompt):
            parser.error("Number of image paths must match number of prompts for Wan2.1")

    # Set default model paths
    if not args.model_path:
        if args.model == "flux":
            args.model_path = "black-forest-labs/FLUX.1-dev"
        elif args.model == "wan2.1":
            args.model_path = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

    # Set default generation parameters
    if args.model == "flux":
        if args.num_inference_steps is None:
            args.num_inference_steps = 28
        if args.guidance_scale is None:
            args.guidance_scale = 3.5
    elif args.model == "wan2.1":
        if args.num_inference_steps is None:
            args.num_inference_steps = 50
        if args.guidance_scale is None:
            args.guidance_scale = 5.0

    logger.info(f"🚀 Starting {args.model.upper()} inference")
    logger.info(f"📝 Prompts: {args.prompt}")

    # Load pipeline
    logger.info(f"📦 Loading {args.model} pipeline...")

    if args.model == "flux":
        pipe = load_flux_pipeline(args.model_path, args.lora_path, args.device)
    elif args.model == "wan2.1":
        pipe = load_wan2_1_pipeline(args.model_path, args.lora_path, args.device)

    logger.info("✅ Pipeline loaded successfully")

    # Generate content
    if args.model == "flux":
        logger.info(f"🎨 Generating {len(args.prompt)} images...")

        images = generate_flux_images(
            pipe=pipe,
            prompts=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            seed=args.seed
        )

        # Save images
        saved_paths = save_images(images, args.output_dir, args.output_prefix)

        logger.info(f"✅ Generated {len(images)} images")
        for path in saved_paths:
            logger.info(f"💾 Saved: {path}")

    elif args.model == "wan2.1":
        logger.info(f"🎬 Generating {len(args.prompt)} videos...")

        videos = generate_wan2_1_videos(
            pipe=pipe,
            image_paths=args.image_path,
            prompts=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )

        # Save videos
        saved_paths = save_videos(videos, args.output_dir, args.output_prefix, args.fps)

        logger.info(f"✅ Generated {len(videos)} videos")
        for path in saved_paths:
            logger.info(f"💾 Saved: {path}")

    print("\n" + "="*60)
    print("🎉 INFERENCE COMPLETED!")
    print("="*60)
    print(f"Model: {args.model.upper()}")
    print(f"Output: {args.output_dir}")
    print(f"Generated: {len(saved_paths)} files")
    print("="*60)


if __name__ == "__main__":
    main()
