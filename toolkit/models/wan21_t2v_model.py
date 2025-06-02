"""
Wan2.1 Text-to-Video model implementation for the diffusion training toolkit
"""

import logging
from typing import Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator

from ..core import BaseModel, ModelConfig
from ..utils import LoRAConfig, QLoRAConfig, DoRAConfig, get_trainable_parameters_info


class Wan21TextToVideoModel(BaseModel):
    """Wan2.1 Text-to-Video model implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = "wan2_1_t2v"
        
    def load_model(self, accelerator: Accelerator) -> Dict[str, Any]:
        """Load Wan2.1 T2V model components"""
        try:
            from diffusers import DiffusionPipeline
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as e:
            raise ImportError(f"Required libraries not available: {e}")
        
        model_path = self.config.pretrained_model_path
        
        # Load the full pipeline first
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            torch_dtype=getattr(torch, self.config.torch_dtype)
        )
        
        # Extract components
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder
        vae = pipeline.vae
        unet = pipeline.unet
        scheduler = pipeline.scheduler
        
        # Setup quantization if needed
        if self.config.use_4bit or self.config.use_8bit:
            quantization_config = self._get_quantization_config()
            # Apply quantization to UNet if needed
            if quantization_config is not None:
                logging.info("Quantization will be applied during LoRA setup")
        
        # Store components
        self.model_components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "vae": vae,
            "unet": unet,
            "scheduler": scheduler
        }
        
        # Freeze non-trainable components
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        # Setup LoRA if enabled
        if self.config.use_lora:
            self.setup_lora()
        
        # Set training mode for UNet
        unet.train()
        
        logging.info("Wan2.1 T2V model loaded successfully")
        param_info = get_trainable_parameters_info(unet)
        logging.info(f"Trainable parameters: {param_info['trainable_parameters']:,} / {param_info['total_parameters']:,} ({param_info['trainable_percentage']:.2f}%)")
        
        return self.model_components
    
    def _get_quantization_config(self):
        """Get quantization configuration"""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("transformers library not available for quantization")
        
        if self.config.use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant
            )
        elif self.config.use_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        
        return None
    
    def setup_lora(self) -> None:
        """Setup LoRA/QLoRA/DoRA for Wan2.1 T2V UNet"""
        try:
            from peft import get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            raise ImportError("PEFT library not available. Install with: pip install peft")
        
        unet = self.model_components["unet"]
        
        # Prepare for k-bit training if using quantization
        if self.config.use_4bit or self.config.use_8bit:
            unet = prepare_model_for_kbit_training(unet)
        
        # Get target modules for video UNet
        target_modules = self.config.lora_target_modules or [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            "conv_in", "conv_out",
            "time_embedding.linear_1", "time_embedding.linear_2"
        ]
        
        # Create LoRA config based on type
        if self.config.lora_type.lower() == "qlora":
            lora_config = QLoRAConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                target_modules=target_modules
            ).to_peft_config()
        elif self.config.lora_type.lower() == "dora":
            lora_config = DoRAConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                target_modules=target_modules
            ).to_peft_config()
        else:  # Standard LoRA
            lora_config = LoRAConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                target_modules=target_modules
            ).to_peft_config()
        
        # Apply LoRA
        unet = get_peft_model(unet, lora_config)
        self.model_components["unet"] = unet
        
        logging.info(f"Applied {self.config.lora_type.upper()} with rank {self.config.lora_rank}")
        
        # Print trainable parameters
        if hasattr(unet, 'print_trainable_parameters'):
            unet.print_trainable_parameters()
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters"""
        return [p for p in self.model_components["unet"].parameters() if p.requires_grad]
    
    def save_model(self, output_dir: str) -> None:
        """Save Wan2.1 T2V model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        unet = self.model_components["unet"]
        
        if self.config.use_lora:
            # Save LoRA weights
            lora_path = output_path / "lora"
            unet.save_pretrained(lora_path)
            logging.info(f"LoRA weights saved to: {lora_path}")
        else:
            # Save full UNet
            unet_path = output_path / "unet"
            unet.save_pretrained(unet_path)
            logging.info(f"Full UNet saved to: {unet_path}")
        
        # Save other components
        for name, component in self.model_components.items():
            if name != "unet" and hasattr(component, 'save_pretrained'):
                component_path = output_path / name
                component.save_pretrained(component_path)
    
    def load_pretrained(self, model_path: str) -> None:
        """Load pretrained Wan2.1 T2V weights"""
        model_path = Path(model_path)
        
        if self.config.use_lora:
            # Load LoRA weights
            lora_path = model_path / "lora"
            if lora_path.exists():
                try:
                    from peft import PeftModel
                    unet = self.model_components["unet"]
                    unet = PeftModel.from_pretrained(unet, lora_path)
                    self.model_components["unet"] = unet
                    logging.info(f"LoRA weights loaded from: {lora_path}")
                except ImportError:
                    raise ImportError("PEFT library not available for loading LoRA weights")
        else:
            # Load full UNet
            unet_path = model_path / "unet"
            if unet_path.exists():
                unet = self.model_components["unet"]
                unet.load_state_dict(torch.load(unet_path / "pytorch_model.bin"))
                logging.info(f"Full UNet loaded from: {unet_path}")
    
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt using CLIP encoder"""
        tokenizer = self.model_components["tokenizer"]
        text_encoder = self.model_components["text_encoder"]
        
        # Tokenize
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode
        with torch.no_grad():
            text_embeddings = text_encoder(
                text_inputs.input_ids.to(text_encoder.device),
                attention_mask=text_inputs.attention_mask.to(text_encoder.device)
            )[0]
        
        return text_embeddings
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video using VAE"""
        vae = self.model_components["vae"]
        
        batch_size, channels, frames, height, width = video.shape
        
        # Reshape for VAE encoding (process frame by frame)
        video = video.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        
        with torch.no_grad():
            latents = vae.encode(video).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        # Reshape back to video format
        latent_channels = latents.shape[1]
        latent_height = latents.shape[2]
        latent_width = latents.shape[3]
        latents = latents.reshape(batch_size, frames, latent_channels, latent_height, latent_width)
        latents = latents.permute(0, 2, 1, 3, 4)
        
        return latents
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video frames using VAE"""
        vae = self.model_components["vae"]
        
        batch_size, channels, frames, height, width = latents.shape
        
        # Scale latents
        latents = latents / vae.config.scaling_factor
        
        # Reshape for VAE decoding (process frame by frame)
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        
        with torch.no_grad():
            video = vae.decode(latents).sample
        
        # Reshape back to video format
        video_channels = video.shape[1]
        video_height = video.shape[2]
        video_width = video.shape[3]
        video = video.reshape(batch_size, frames, video_channels, video_height, video_width)
        video = video.permute(0, 2, 1, 3, 4)
        
        return video
    
    def prepare_video_latents(self, batch_size: int, num_frames: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Prepare random latents for video generation"""
        vae = self.model_components["vae"]
        
        # Calculate latent dimensions
        latent_height = height // vae.config.scaling_factor
        latent_width = width // vae.config.scaling_factor
        latent_channels = vae.config.latent_channels
        
        # Generate random latents
        shape = (batch_size, latent_channels, num_frames, latent_height, latent_width)
        latents = torch.randn(shape, device=device, dtype=torch.float32)
        
        return latents
