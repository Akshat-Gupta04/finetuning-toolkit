"""
Unified training script for both Wan2.1 and FLUX models
Supports variable-size training, LoRA, QLoRA, and production features
"""

import os
import argparse
import logging
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler

from config_manager import get_config_manager, load_config
from dataset_variable_size import VariableSizeImageVideoDataset, VariableSizeFluxDataset
from data_collator import create_variable_size_dataloader
from lora_utils import validate_lora_config, get_lora_config_summary


def setup_environment():
    """Setup production environment"""
    config_manager = get_config_manager()
    config_manager.setup_all()

    return config_manager


def load_models(config: Dict[str, Any], model_type: str, accelerator: Accelerator):
    """Load models based on type with LoRA/QLoRA support"""

    # Validate LoRA configuration
    if config["model"].get("use_lora", False):
        if not validate_lora_config(config, model_type):
            raise ValueError("Invalid LoRA configuration")

        # Log LoRA configuration
        lora_summary = get_lora_config_summary(config, model_type)
        logging.info(f"LoRA Configuration: {lora_summary}")

    if model_type == "wan2_1_i2v":
        from train_wan2_1 import load_models as load_wan2_1_models
        return load_wan2_1_models(config, accelerator)
    elif model_type == "flux":
        from train_flux import load_models as load_flux_models
        return load_flux_models(config, accelerator)
    elif model_type == "wan2_1_t2v":
        from train_wan2_1_t2v import load_wan2_1_t2v_models
        return load_wan2_1_t2v_models(config, accelerator)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: flux, wan2_1_i2v, wan2_1_t2v")


def create_dataset(config: Dict[str, Any], model_type: str, variable_size: bool = True):
    """Create dataset based on model type and configuration"""
    dataset_config = config["dataset"]
    training_config = config["training"]

    if model_type == "wan2_1_i2v":
        if variable_size:
            dataset = VariableSizeImageVideoDataset(
                data_dir=dataset_config["train_data_dir"],
                image_column=dataset_config["image_column"],
                video_column=dataset_config["video_column"],
                caption_column=dataset_config["caption_column"],
                max_width=training_config.get("max_width", 1280),
                max_height=training_config.get("max_height", 720),
                num_frames=training_config.get("num_frames", 81),
                resize_mode=dataset_config.get("resize_mode", "pad"),
                bucket_sizes=training_config.get("bucket_sizes")
            )
        else:
            from dataset import ImageVideoDataset
            dataset = ImageVideoDataset(
                data_dir=dataset_config["train_data_dir"],
                image_column=dataset_config["image_column"],
                video_column=dataset_config["video_column"],
                caption_column=dataset_config["caption_column"],
                width=training_config["width"],
                height=training_config["height"],
                num_frames=training_config["num_frames"]
            )

    elif model_type == "flux":
        if variable_size:
            dataset = VariableSizeFluxDataset(
                data_dir=dataset_config["train_data_dir"],
                image_column=dataset_config["image_column"],
                caption_column=dataset_config["caption_column"],
                max_resolution=training_config.get("max_resolution", 1024),
                resize_mode=dataset_config.get("resize_mode", "pad"),
                bucket_sizes=training_config.get("bucket_sizes")
            )
        else:
            from dataset_flux import FluxTextToImageDataset
            dataset = FluxTextToImageDataset(
                data_dir=dataset_config["train_data_dir"],
                image_column=dataset_config["image_column"],
                caption_column=dataset_config["caption_column"],
                resolution=training_config["resolution"]
            )
    elif model_type == "wan2_1_t2v":
        from train_wan2_1_t2v import create_wan2_1_t2v_dataset
        dataset = create_wan2_1_t2v_dataset(config, variable_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: flux, wan2_1_i2v, wan2_1_t2v")

    return dataset


def create_dataloader(dataset, config: Dict[str, Any], variable_size: bool = True):
    """Create dataloader with appropriate settings"""
    training_config = config["training"]

    if variable_size:
        return create_variable_size_dataloader(
            dataset=dataset,
            batch_size=training_config["train_batch_size"],
            shuffle=True,
            num_workers=training_config.get("dataloader_num_workers", 4),
            pin_memory=training_config.get("pin_memory", True),
            use_bucket_sampler=training_config.get("use_bucket_sampler", True)
        )
    else:
        return DataLoader(
            dataset,
            batch_size=training_config["train_batch_size"],
            shuffle=True,
            num_workers=training_config.get("dataloader_num_workers", 4),
            pin_memory=training_config.get("pin_memory", True)
        )


def setup_optimizer_and_scheduler(model, config: Dict[str, Any]):
    """Setup optimizer and learning rate scheduler"""
    training_config = config["training"]

    # Optimizer
    if training_config.get("use_8bit_adam", False):
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=(training_config["adam_beta1"], training_config["adam_beta2"]),
        weight_decay=training_config["adam_weight_decay"],
        eps=training_config["adam_epsilon"]
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        training_config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=training_config["lr_warmup_steps"],
        num_training_steps=training_config["max_train_steps"],
        num_cycles=training_config.get("lr_num_cycles", 1)
    )

    return optimizer, lr_scheduler


def compute_loss(models: Dict[str, Any], batch: Dict[str, Any], config: Dict[str, Any], model_type: str, accelerator: Accelerator):
    """Compute loss based on model type"""
    if model_type == "wan2_1_i2v":
        from train_wan2_1 import compute_loss as compute_wan2_1_loss
        return compute_wan2_1_loss(models, batch, config, accelerator)
    elif model_type == "flux":
        from train_flux import compute_loss as compute_flux_loss
        return compute_flux_loss(models, batch, config, accelerator)
    elif model_type == "wan2_1_t2v":
        from train_wan2_1_t2v import compute_wan2_1_t2v_loss
        return compute_wan2_1_t2v_loss(models, batch, config, accelerator)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: flux, wan2_1_i2v, wan2_1_t2v")


def train_model(
    config: Dict[str, Any],
    model_type: str,
    variable_size: bool = True,
    resume_from_checkpoint: Optional[str] = None
):
    """Main training function"""

    # Setup environment
    setup_environment()

    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=config["training"]["output_dir"],
        logging_dir=config["training"]["logging_dir"]
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with=config["training"].get("report_to"),
        project_config=accelerator_project_config
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # Set seed
    if config["training"].get("seed") is not None:
        set_seed(config["training"]["seed"])

    # Create output directory
    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    # Load models
    logger.info(f"Loading {model_type} models...")
    models = load_models(config, model_type, accelerator)

    # Create dataset
    logger.info("Creating dataset...")
    dataset = create_dataset(config, model_type, variable_size)

    # Create dataloader
    logger.info("Creating dataloader...")
    dataloader = create_dataloader(dataset, config, variable_size)

    # Get main model for training
    main_model_key = "transformer" if model_type in ["wan2.1", "flux"] else "unet"
    main_model = models[main_model_key]

    # Setup optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(main_model, config)

    # Prepare with accelerator
    main_model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        main_model, optimizer, dataloader, lr_scheduler
    )

    # Update models dict
    models[main_model_key] = main_model

    # Move other models to device
    for key, model in models.items():
        if key != main_model_key and hasattr(model, 'to'):
            model.to(accelerator.device)

    # Training parameters
    max_train_steps = config["training"]["max_train_steps"]
    logging_steps = config["training"].get("logging_steps", 10)
    save_steps = config["training"].get("save_steps", 500)

    # Resume from checkpoint
    global_step = 0
    if resume_from_checkpoint:
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(resume_from_checkpoint.split("-")[-1])

    # Setup W&B if configured
    if config["training"].get("report_to") == "wandb" and accelerator.is_main_process:
        try:
            import wandb
            wandb.init(
                project=config.get("wandb", {}).get("project_name", "diffusion-training"),
                name=config.get("wandb", {}).get("run_name", f"{model_type}-training"),
                config=config
            )
        except ImportError:
            logger.warning("wandb not available")

    # Training loop
    logger.info(f"Starting training for {max_train_steps} steps")

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process
    )

    main_model.train()

    for step, batch in enumerate(dataloader):
        if global_step >= max_train_steps:
            break

        with accelerator.accumulate(main_model):
            # Compute loss
            loss = compute_loss(models, batch, config, model_type, accelerator)

            # Backward pass
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(main_model.parameters(), config["training"]["max_grad_norm"])

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Update progress
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            # Logging
            if global_step % logging_steps == 0:
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step
                }

                logger.info(f"Step {global_step}: loss={loss.detach().item():.4f}, lr={lr_scheduler.get_last_lr()[0]:.2e}")

                # Log to W&B
                try:
                    import wandb
                    if wandb.run is not None and accelerator.is_main_process:
                        wandb.log(logs, step=global_step)
                except:
                    pass

            # Save checkpoint
            if global_step % save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(config["training"]["output_dir"], f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint: {save_path}")

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_model = accelerator.unwrap_model(main_model)

        if config["model"].get("use_lora", False):
            final_model.save_pretrained(os.path.join(config["training"]["output_dir"], "lora"))
        else:
            final_model.save_pretrained(os.path.join(config["training"]["output_dir"], main_model_key))

        logger.info(f"Training completed. Model saved to {config['training']['output_dir']}")

    accelerator.end_training()
    return global_step


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Unified diffusion model training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--model_type", type=str, required=True, choices=["flux", "wan2_1_i2v", "wan2_1_t2v"], help="Model type")
    parser.add_argument("--variable_size", action="store_true", help="Enable variable-size training")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, help="Output directory override")
    parser.add_argument("--logging_dir", type=str, help="Logging directory override")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.logging_dir:
        config["training"]["logging_dir"] = args.logging_dir

    # Train model
    final_step = train_model(
        config=config,
        model_type=args.model_type,
        variable_size=args.variable_size,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    print(f"✅ Training completed at step {final_step}")


if __name__ == "__main__":
    main()
