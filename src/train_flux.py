"""
FLUX.1-dev Text-to-Image Model Finetuning Script
Supports LoRA, QLoRA, and Dreambooth training
"""

import os
import sys
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import wandb

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.utils import export_to_gif
from diffusers.optimization import get_scheduler

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

from omegaconf import OmegaConf
from dataset_flux import FluxTextToImageDataset, FluxDreamboothDataset

# Setup logging
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune FLUX.1-dev text-to-image model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/flux_lora_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="Logging directory (overrides config)"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help="Reporting service (wandb, tensorboard, etc.)"
    )

    args = parser.parse_args()
    return args


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def setup_logging(logging_dir: str, accelerator: Accelerator):
    """Setup logging configuration"""
    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    else:
        transformers.utils.logging.set_verbosity_error()
        logging.basicConfig(level=logging.ERROR)


def load_models(config: Dict[str, Any], accelerator: Accelerator):
    """Load all FLUX model components with quantization support"""
    model_config = config["model"]

    # Setup quantization config for QLoRA
    quantization_config = None
    if model_config.get("use_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
        )
    elif model_config.get("use_8bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load tokenizers
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        subfolder="tokenizer",
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
    )

    t5_tokenizer = T5TokenizerFast.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        subfolder="tokenizer_2",
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
    )

    # Load text encoders
    text_encoder = CLIPTextModel.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        subfolder="text_encoder",
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
        torch_dtype=torch.float32,
    )

    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        subfolder="text_encoder_2",
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
        torch_dtype=torch.float32,
    )

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        subfolder="vae",
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
        torch_dtype=torch.float32,
    )

    # Load transformer (main model to finetune)
    transformer_dtype = torch.bfloat16
    if quantization_config is not None:
        transformer_dtype = torch.float32

    transformer = FluxTransformer2DModel.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        subfolder="transformer",
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
        torch_dtype=transformer_dtype,
        quantization_config=quantization_config,
        device_map=model_config.get("device_map", None),
    )

    # Prepare model for k-bit training if using quantization
    if quantization_config is not None:
        transformer = prepare_model_for_kbit_training(transformer)

    # Setup LoRA/QLoRA if enabled
    if model_config.get("use_lora", False):
        lora_type = model_config.get("lora_type", "lora").lower()

        # Base LoRA configuration
        lora_config_kwargs = {
            "r": model_config.get("lora_rank", 16),
            "lora_alpha": model_config.get("lora_alpha", 16),
            "target_modules": model_config.get("lora_target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
            "lora_dropout": model_config.get("lora_dropout", 0.1),
            "bias": "none",
            "task_type": TaskType.DIFFUSION,
        }

        # Create LoRA config
        lora_config = LoraConfig(**lora_config_kwargs)

        # Apply LoRA to model
        transformer = get_peft_model(transformer, lora_config)

        # Print trainable parameters
        if hasattr(transformer, 'print_trainable_parameters'):
            transformer.print_trainable_parameters()
        else:
            total_params = sum(p.numel() for p in transformer.parameters())
            trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        logger.info(f"Using {lora_type.upper()} with rank {model_config.get('lora_rank', 16)}")
        if quantization_config is not None:
            logger.info(f"Quantization: {'4-bit' if model_config.get('use_4bit') else '8-bit'}")

    # Freeze components that shouldn't be trained
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Enable training mode for transformer
    transformer.train()

    # Setup noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        subfolder="scheduler",
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
    )

    return {
        "clip_tokenizer": clip_tokenizer,
        "t5_tokenizer": t5_tokenizer,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "vae": vae,
        "transformer": transformer,
        "noise_scheduler": noise_scheduler,
    }


def encode_prompt(clip_tokenizer, t5_tokenizer, text_encoder, text_encoder_2, prompt: str, max_length: int = 512):
    """Encode text prompt using both CLIP and T5 encoders"""
    # CLIP encoding
    clip_inputs = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        clip_embeddings = text_encoder(
            clip_inputs.input_ids.to(text_encoder.device),
            attention_mask=clip_inputs.attention_mask.to(text_encoder.device),
        )[0]

    # T5 encoding
    t5_inputs = t5_tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        t5_embeddings = text_encoder_2(
            t5_inputs.input_ids.to(text_encoder_2.device),
            attention_mask=t5_inputs.attention_mask.to(text_encoder_2.device),
        )[0]

    return clip_embeddings, t5_embeddings


def compute_loss(
    models: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    accelerator: Accelerator,
):
    """Compute training loss for FLUX"""
    vae = models["vae"]
    transformer = models["transformer"]
    noise_scheduler = models["noise_scheduler"]
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]
    clip_tokenizer = models["clip_tokenizer"]
    t5_tokenizer = models["t5_tokenizer"]

    # Get batch data
    images = batch["pixel_values"]  # (B, C, H, W)
    captions = batch["caption"]

    batch_size = images.shape[0]

    # Encode images with VAE
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Encode text prompts
    clip_embeddings_list = []
    t5_embeddings_list = []

    for caption in captions:
        clip_emb, t5_emb = encode_prompt(
            clip_tokenizer, t5_tokenizer, text_encoder, text_encoder_2,
            caption, config["dataset"]["max_sequence_length"]
        )
        clip_embeddings_list.append(clip_emb)
        t5_embeddings_list.append(t5_emb)

    clip_embeddings = torch.cat(clip_embeddings_list, dim=0)
    t5_embeddings = torch.cat(t5_embeddings_list, dim=0)

    # Sample noise
    noise = torch.randn_like(latents)

    # Sample timesteps
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (batch_size,), device=latents.device
    ).long()

    # Add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Predict noise
    model_pred = transformer(
        noisy_latents,
        timesteps,
        encoder_hidden_states=t5_embeddings,
        pooled_projections=clip_embeddings,
        return_dict=False,
    )[0]

    # Compute loss
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss


def validate_model(
    models: Dict[str, Any],
    config: Dict[str, Any],
    accelerator: Accelerator,
    step: int,
):
    """Run validation and generate sample images"""
    if not accelerator.is_main_process:
        return

    validation_config = config["validation"]

    # Create pipeline for inference
    pipeline = FluxPipeline(
        vae=models["vae"],
        text_encoder=models["text_encoder"],
        text_encoder_2=models["text_encoder_2"],
        tokenizer=models["clip_tokenizer"],
        tokenizer_2=models["t5_tokenizer"],
        transformer=accelerator.unwrap_model(models["transformer"]),
        scheduler=models["noise_scheduler"],
    )
    pipeline.to(accelerator.device)

    # Generate validation images
    validation_prompts = validation_config.get("validation_prompts", ["A beautiful landscape"])

    output_dir = Path(config["training"]["output_dir"])
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(exist_ok=True)

    for i, prompt in enumerate(validation_prompts):
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                height=config["training"]["height"],
                width=config["training"]["width"],
                guidance_scale=validation_config["guidance_scale"],
                num_inference_steps=validation_config["num_inference_steps"],
                generator=torch.Generator(device=accelerator.device).manual_seed(config["training"]["seed"]),
            ).images[0]

        # Save validation image
        image_path = validation_dir / f"validation_step_{step}_prompt_{i}.png"
        image.save(image_path)

        # Log to wandb if enabled
        if config["training"]["report_to"] == "wandb":
            wandb.log({
                f"validation_image_{i}": wandb.Image(image, caption=prompt),
            }, step=step)

    logger.info(f"Saved validation images to: {validation_dir}")


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.logging_dir:
        config["training"]["logging_dir"] = args.logging_dir
    if args.mixed_precision:
        config["training"]["mixed_precision"] = args.mixed_precision
    if args.report_to:
        config["training"]["report_to"] = args.report_to
    if args.resume_from_checkpoint:
        config["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint

    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=config["training"]["output_dir"],
        logging_dir=config["training"]["logging_dir"],
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with=config["training"]["report_to"],
        project_config=accelerator_project_config,
    )

    # Setup logging
    setup_logging(config["training"]["logging_dir"], accelerator)

    # Set seed
    if config["training"]["seed"] is not None:
        set_seed(config["training"]["seed"])

    # Create output directory
    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    # Initialize wandb
    if accelerator.is_main_process and config["training"]["report_to"] == "wandb":
        wandb.init(
            project=config["wandb"]["project_name"],
            name=config["wandb"]["run_name"],
            tags=config["wandb"]["tags"],
            notes=config["wandb"]["notes"],
            config=config,
        )

    # Load models
    logger.info("Loading models...")
    models = load_models(config, accelerator)

    # Create datasets
    logger.info("Creating datasets...")

    # Determine training type
    training_type = config.get("flux", {}).get("training_type", "text_to_image")

    if training_type == "dreambooth":
        train_dataset = FluxDreamboothDataset(
            instance_data_dir=config["dataset"]["train_data_dir"],
            class_data_dir=config["dataset"].get("validation_data_dir"),
            instance_prompt=config["flux"]["instance_prompt"],
            class_prompt=config["flux"]["class_prompt"],
            resolution=config["training"]["resolution"],
            center_crop=config["dataset"].get("center_crop", True),
            random_flip=config["dataset"].get("random_flip", 0.0),
        )
    else:
        train_dataset = FluxTextToImageDataset(
            data_dir=config["dataset"]["train_data_dir"],
            image_column=config["dataset"]["image_column"],
            caption_column=config["dataset"]["caption_column"],
            resolution=config["training"]["resolution"],
            center_crop=config["dataset"].get("center_crop", True),
            random_flip=config["dataset"].get("random_flip", 0.0),
            max_sequence_length=config["dataset"]["max_sequence_length"],
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        num_workers=config["training"]["dataloader_num_workers"],
        pin_memory=config["training"].get("pin_memory", True),
    )

    # Setup optimizer
    if config["training"]["use_8bit_adam"]:
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        models["transformer"].parameters(),
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        weight_decay=config["training"]["adam_weight_decay"],
        eps=config["training"]["adam_epsilon"],
    )

    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"],
        num_training_steps=config["training"]["max_train_steps"],
        num_cycles=config["training"].get("lr_num_cycles", 1),
    )

    # Prepare everything with accelerator
    models["transformer"], optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        models["transformer"], optimizer, train_dataloader, lr_scheduler
    )

    # Move other models to device
    models["vae"].to(accelerator.device)
    models["text_encoder"].to(accelerator.device)
    models["text_encoder_2"].to(accelerator.device)
    models["noise_scheduler"].to(accelerator.device)

    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["training"]["gradient_accumulation_steps"])
    max_train_steps = config["training"]["max_train_steps"]
    num_train_epochs = config["training"]["num_train_epochs"]

    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    logger.info(f"***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config['training']['train_batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {config['training']['train_batch_size'] * accelerator.num_processes * config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Gradient Accumulation steps = {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Training loop
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint if specified
    if config["training"]["resume_from_checkpoint"]:
        accelerator.load_state(config["training"]["resume_from_checkpoint"])
        global_step = int(config["training"]["resume_from_checkpoint"].split("-")[-1])
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        models["transformer"].train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(models["transformer"]):
                # Compute loss
                loss = compute_loss(models, batch, config, accelerator)

                # Backward pass
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(models["transformer"].parameters(), config["training"]["max_grad_norm"])

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Logging
                if global_step % config["training"]["logging_steps"] == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "epoch": epoch,
                    }

                    if config["training"]["report_to"] == "wandb" and accelerator.is_main_process:
                        wandb.log(logs, step=global_step)

                    logger.info(f"Step {global_step}: loss={loss.detach().item():.4f}, lr={lr_scheduler.get_last_lr()[0]:.2e}")

                # Validation
                if global_step % config["training"]["validation_steps"] == 0:
                    validate_model(models, config, accelerator, global_step)

                # Save checkpoint
                if global_step % config["training"]["checkpointing_steps"] == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(config["training"]["output_dir"], f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint: {save_path}")

                # Check if we've reached max steps
                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(models["transformer"])

        if config["model"].get("use_lora", False):
            # Save LoRA weights
            transformer.save_pretrained(os.path.join(config["training"]["output_dir"], "lora"))
        else:
            # Save full model
            transformer.save_pretrained(os.path.join(config["training"]["output_dir"], "transformer"))

        logger.info(f"Training completed. Model saved to {config['training']['output_dir']}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
