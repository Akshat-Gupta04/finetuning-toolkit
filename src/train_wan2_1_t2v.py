"""
Wan2.1 Text-to-Video model training with LoRA/QLoRA support
Supports Wan2.1 T2V models for text-to-video generation
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging
from pathlib import Path

try:
    from diffusers import DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers not available. Wan2.1 T2V training will be disabled.")

from lora_utils import prepare_model_for_lora, load_quantized_model, validate_lora_config


def load_wan2_1_t2v_models(config: Dict[str, Any], accelerator) -> Dict[str, Any]:
    """Load Wan2.1 text-to-video models for training"""
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("Diffusers not available. Cannot load Wan2.1 T2V models.")

    model_config = config["model"]
    model_path = model_config.get("pretrained_model_name_or_path", "Wan-AI/Wan2.1-T2V-14B-720P-Diffusers")

    logging.info(f"Loading Wan2.1 T2V models from {model_path}")

    # Load quantization config if needed
    quantization_config = load_quantized_model(model_path, config)

    try:
        # Load the pipeline first to get components
        if quantization_config:
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                **quantization_config
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            )

        # Extract components
        unet = pipe.unet
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        scheduler = pipe.scheduler
        vae = pipe.vae

        # Prepare UNet for LoRA training
        if config["model"].get("use_lora", False):
            if validate_lora_config(config, "wan2_1_t2v"):
                unet = prepare_model_for_lora(unet, config, "wan2_1_t2v")
            else:
                logging.error("LoRA configuration validation failed")

        # Set models to appropriate modes
        unet.train()
        text_encoder.eval()
        vae.eval()

        # Disable gradients for non-trainable models
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)

        # Enable gradient checkpointing if configured
        if config["training"].get("gradient_checkpointing", False):
            unet.enable_gradient_checkpointing()
            if hasattr(text_encoder, "gradient_checkpointing_enable"):
                text_encoder.gradient_checkpointing_enable()

        models = {
            "unet": unet,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "vae": vae,
            "pipeline": pipe
        }

        logging.info("Wan2.1 T2V models loaded successfully")
        return models

    except Exception as e:
        logging.error(f"Failed to load Wan2.1 T2V models: {e}")
        raise


def compute_wan2_1_t2v_loss(models: Dict[str, Any], batch: Dict[str, Any], config: Dict[str, Any], accelerator) -> torch.Tensor:
    """Compute loss for Wan2.1 text-to-video training"""
    unet = models["unet"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    scheduler = models["scheduler"]
    vae = models["vae"]

    # Get batch data
    videos = batch.get("video")
    captions = batch.get("caption", batch.get("text"))

    if videos is None:
        raise ValueError("No video data found in batch")
    if captions is None:
        raise ValueError("No caption/text data found in batch")

    batch_size, num_frames = videos.shape[:2]
    device = accelerator.device

    # Encode videos to latent space
    videos = videos.to(device, dtype=torch.bfloat16)

    # Reshape for VAE encoding: [B*F, C, H, W]
    videos_reshaped = videos.view(-1, *videos.shape[2:])

    with torch.no_grad():
        # Encode frames to latents
        latents = vae.encode(videos_reshaped).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Reshape back to video format: [B, F, C, H, W]
        latents = latents.view(batch_size, num_frames, *latents.shape[1:])

    # Sample noise
    noise = torch.randn_like(latents)

    # Sample timesteps
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (batch_size,), device=device
    ).long()

    # Add noise to latents
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # Encode text
    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        text_embeddings = text_encoder(
            text_inputs.input_ids.to(device)
        )[0]

    # Predict noise
    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=text_embeddings
    ).sample

    # Compute loss
    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss


def create_wan2_1_t2v_dataset(config: Dict[str, Any], variable_size: bool = True):
    """Create Wan2.1 text-to-video dataset"""
    from dataset_variable_size import VariableSizeText2VideoDataset
    from dataset import Text2VideoDataset

    dataset_config = config["dataset"]
    training_config = config["training"]

    if variable_size:
        dataset = VariableSizeText2VideoDataset(
            data_dir=dataset_config["train_data_dir"],
            video_column=dataset_config.get("video_column", "video"),
            caption_column=dataset_config.get("caption_column", "caption"),
            max_width=training_config.get("max_width", 1280),
            max_height=training_config.get("max_height", 720),
            num_frames=training_config.get("num_frames", 81),
            frame_rate=training_config.get("frame_rate", 16),
            resize_mode=dataset_config.get("resize_mode", "pad"),
            bucket_sizes=training_config.get("bucket_sizes")
        )
    else:
        dataset = Text2VideoDataset(
            data_dir=dataset_config["train_data_dir"],
            video_column=dataset_config.get("video_column", "video"),
            caption_column=dataset_config.get("caption_column", "caption"),
            width=training_config.get("width", 1280),
            height=training_config.get("height", 720),
            num_frames=training_config.get("num_frames", 81),
            frame_rate=training_config.get("frame_rate", 16)
        )

    return dataset


def setup_wan2_1_t2v_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup Wan2.1 T2V specific training configuration"""
    training_config = config["training"].copy()

    # Wan2.1 T2V specific settings
    training_config.update({
        "num_frames": config["model_overrides"]["wan2_1_t2v"].get("num_frames", 81),
        "frame_rate": config["model_overrides"]["wan2_1_t2v"].get("frame_rate", 16),
        "guidance_scale": config["model_overrides"]["wan2_1_t2v"].get("guidance_scale", 7.5),
        "num_inference_steps": config["model_overrides"]["wan2_1_t2v"].get("num_inference_steps", 50)
    })

    # Adjust batch size for video training (videos use more memory)
    if "hardware_optimizations" in config:
        hw_opts = config["hardware_optimizations"]
        if hw_opts.get("enable_large_batch_training", False):
            batch_sizes = hw_opts.get("recommended_batch_sizes", {})
            resolution = training_config.get("max_height", 720)

            if resolution >= 1080:
                training_config["train_batch_size"] = batch_sizes.get("wan2_1_t2v_1080p", 1)
            else:
                training_config["train_batch_size"] = batch_sizes.get("wan2_1_t2v_720p", 2)

    return training_config


def save_wan2_1_t2v_model(models: Dict[str, Any], output_dir: str, config: Dict[str, Any]) -> None:
    """Save Wan2.1 T2V model"""
    from lora_utils import save_lora_model

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unet = models["unet"]

    # Save LoRA model if enabled
    if config["model"].get("use_lora", False):
        lora_path = output_path / "lora"
        save_lora_model(unet, str(lora_path))
    else:
        # Save full model
        unet.save_pretrained(output_path / "unet")

    # Save other components
    models["text_encoder"].save_pretrained(output_path / "text_encoder")
    models["tokenizer"].save_pretrained(output_path / "tokenizer")

    # Save scheduler config
    models["scheduler"].save_pretrained(output_path / "scheduler")

    logging.info(f"Wan2.1 T2V model saved to {output_dir}")


def load_wan2_1_t2v_for_inference(model_path: str, lora_path: Optional[str] = None) -> DiffusionPipeline:
    """Load Wan2.1 T2V model for inference"""
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("Diffusers not available. Cannot load Wan2.1 T2V model for inference.")

    try:
        # Load base pipeline
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )

        # Load LoRA weights if provided
        if lora_path:
            from lora_utils import load_lora_model
            pipe.unet = load_lora_model(pipe.unet, lora_path)

        return pipe

    except Exception as e:
        logging.error(f"Failed to load Wan2.1 T2V model for inference: {e}")
        raise


def validate_wan2_1_t2v_batch(batch: Dict[str, Any]) -> bool:
    """Validate Wan2.1 T2V training batch"""
    required_keys = ["video"]
    caption_keys = ["caption", "text"]

    for key in required_keys:
        if key not in batch:
            logging.error(f"Missing required key in batch: {key}")
            return False

    # Check for caption or text
    has_caption = any(key in batch for key in caption_keys)
    if not has_caption:
        logging.error(f"Missing caption data. Expected one of: {caption_keys}")
        return False

    videos = batch["video"]
    captions = batch.get("caption", batch.get("text"))

    # Check video shape
    if len(videos.shape) != 5:  # [B, F, C, H, W]
        logging.error(f"Invalid video shape: {videos.shape}. Expected [B, F, C, H, W]")
        return False

    # Check captions
    if not isinstance(captions, (list, tuple)):
        logging.error("Captions must be a list or tuple of strings")
        return False

    if len(captions) != videos.shape[0]:
        logging.error(f"Batch size mismatch: {len(captions)} captions, {videos.shape[0]} videos")
        return False

    return True
