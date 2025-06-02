"""
FLUX trainer implementation for the diffusion training toolkit
"""

import os
import math
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..core import BaseTrainer, BaseDataset, TrainingConfig
from ..models.flux_model import FluxModel
from ..utils import format_time, get_gpu_memory_info


class FluxTrainer(BaseTrainer):
    """Trainer for FLUX text-to-image model"""
    
    def __init__(self, model: FluxModel, config: TrainingConfig):
        super().__init__(model, config)
        self.model_type = "flux"
        
    def setup_training(self, dataset: BaseDataset) -> None:
        """Setup training components"""
        # Setup accelerator
        self.setup_accelerator()
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=dataset.collate_fn
        )
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Setup scheduler
        self.setup_scheduler()
        
        # Prepare with accelerator
        (
            self.model.model_components["transformer"],
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.model.model_components["transformer"],
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler
        )
        
        # Move other components to device
        for name, component in self.model.model_components.items():
            if name != "transformer" and hasattr(component, 'to'):
                component.to(self.accelerator.device)
        
        logging.info("FLUX trainer setup completed")
    
    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute FLUX training loss"""
        # Get model components
        vae = self.model.model_components["vae"]
        transformer = self.model.model_components["transformer"]
        scheduler = self.model.model_components["scheduler"]
        text_encoder = self.model.model_components["text_encoder"]
        text_encoder_2 = self.model.model_components["text_encoder_2"]
        clip_tokenizer = self.model.model_components["clip_tokenizer"]
        t5_tokenizer = self.model.model_components["t5_tokenizer"]
        
        # Get batch data
        images = batch["pixel_values"]
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
            clip_emb, t5_emb = self.model.encode_prompt(caption, max_length=512)
            clip_embeddings_list.append(clip_emb)
            t5_embeddings_list.append(t5_emb)
        
        clip_embeddings = torch.cat(clip_embeddings_list, dim=0)
        t5_embeddings = torch.cat(t5_embeddings_list, dim=0)
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (batch_size,), device=latents.device
        ).long()
        
        # Add noise to latents
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        model_pred = transformer(
            noisy_latents,
            timesteps,
            encoder_hidden_states=t5_embeddings,
            pooled_projections=clip_embeddings,
            return_dict=False
        )[0]
        
        # Compute loss
        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss
    
    def validation_step(self) -> Dict[str, Any]:
        """Run validation step"""
        if not self.accelerator.is_main_process:
            return {}
        
        # Generate sample images
        validation_prompts = [
            "A beautiful landscape with mountains and lakes",
            "A futuristic city at sunset",
            "A cute cat sitting on a windowsill"
        ]
        
        try:
            from diffusers import FluxPipeline
            
            # Create pipeline
            pipeline = FluxPipeline(
                vae=self.model.model_components["vae"],
                text_encoder=self.model.model_components["text_encoder"],
                text_encoder_2=self.model.model_components["text_encoder_2"],
                tokenizer=self.model.model_components["clip_tokenizer"],
                tokenizer_2=self.model.model_components["t5_tokenizer"],
                transformer=self.accelerator.unwrap_model(self.model.model_components["transformer"]),
                scheduler=self.model.model_components["scheduler"]
            )
            pipeline.to(self.accelerator.device)
            
            # Generate validation images
            validation_dir = Path(self.config.output_dir) / "validation"
            validation_dir.mkdir(exist_ok=True)
            
            validation_results = {}
            
            for i, prompt in enumerate(validation_prompts):
                with torch.no_grad():
                    image = pipeline(
                        prompt=prompt,
                        height=512,
                        width=512,
                        guidance_scale=7.5,
                        num_inference_steps=20,
                        generator=torch.Generator(device=self.accelerator.device).manual_seed(42)
                    ).images[0]
                
                # Save image
                image_path = validation_dir / f"validation_step_{self.global_step}_prompt_{i}.png"
                image.save(image_path)
                
                validation_results[f"validation_image_{i}"] = str(image_path)
            
            return validation_results
            
        except Exception as e:
            logging.warning(f"Validation failed: {e}")
            return {}
    
    def train(self, dataset: BaseDataset, resume_from_checkpoint: Optional[str] = None) -> None:
        """Main training loop"""
        # Calculate training steps
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        
        if self.config.max_train_steps is None:
            max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        else:
            max_train_steps = self.config.max_train_steps
        
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        
        # Resume from checkpoint if specified
        first_epoch = 0
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            first_epoch = self.global_step // num_update_steps_per_epoch
        
        # Log training info
        logging.info("***** Running FLUX Training *****")
        logging.info(f"  Num examples = {len(dataset)}")
        logging.info(f"  Num epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size = {self.config.train_batch_size}")
        logging.info(f"  Total train batch size = {self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps}")
        logging.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
        
        # Progress bar
        progress_bar = tqdm(
            range(0, max_train_steps),
            initial=self.global_step,
            desc="Training",
            disable=not self.accelerator.is_local_main_process
        )
        
        # Training loop
        for epoch in range(first_epoch, num_train_epochs):
            self.model.model_components["transformer"].train()
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model.model_components["transformer"]):
                    # Compute loss
                    loss = self.compute_loss(batch)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.get_trainable_parameters(),
                            self.config.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Update progress
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        gpu_info = get_gpu_memory_info()
                        
                        logs = {
                            "loss": loss.detach().item(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "step": self.global_step,
                            "epoch": epoch,
                            "gpu_memory_allocated_mb": gpu_info["allocated_mb"],
                            "gpu_memory_free_mb": gpu_info["free_mb"]
                        }
                        
                        self.log_metrics(logs, self.global_step)
                        
                        logging.info(
                            f"Step {self.global_step}: loss={loss.detach().item():.4f}, "
                            f"lr={self.lr_scheduler.get_last_lr()[0]:.2e}, "
                            f"gpu_mem={gpu_info['allocated_mb']:.0f}MB"
                        )
                    
                    # Validation
                    if self.global_step % self.config.validation_steps == 0:
                        validation_results = self.validation_step()
                        if validation_results:
                            self.log_metrics(validation_results, self.global_step)
                    
                    # Save checkpoint
                    if self.global_step % self.config.checkpointing_steps == 0:
                        self.save_checkpoint(self.global_step)
                    
                    # Check if we've reached max steps
                    if self.global_step >= max_train_steps:
                        break
            
            if self.global_step >= max_train_steps:
                break
        
        # Final save
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            final_model_dir = os.path.join(self.config.output_dir, "final_model")
            self.model.save_model(final_model_dir)
            logging.info(f"Final model saved to: {final_model_dir}")
        
        self.accelerator.end_training()
        logging.info("FLUX training completed!")
