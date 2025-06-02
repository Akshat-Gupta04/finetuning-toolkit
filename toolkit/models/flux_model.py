"""
FLUX model implementation for the diffusion training toolkit
"""

import logging
from typing import Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator

from ..core import BaseModel, ModelConfig
from ..utils import LoRAConfig, QLoRAConfig, DoRAConfig, get_trainable_parameters_info


class FluxModel(BaseModel):
    """FLUX text-to-image model implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = "flux"
        
    def load_model(self, accelerator: Accelerator) -> Dict[str, Any]:
        """Load FLUX model components"""
        try:
            from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
            from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
        except ImportError as e:
            raise ImportError(f"Required libraries not available: {e}")
        
        model_path = self.config.pretrained_model_path
        
        # Load tokenizers
        clip_tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            revision=self.config.revision,
            cache_dir=self.config.cache_dir
        )
        
        t5_tokenizer = T5TokenizerFast.from_pretrained(
            model_path,
            subfolder="tokenizer_2", 
            revision=self.config.revision,
            cache_dir=self.config.cache_dir
        )
        
        # Load text encoders
        text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            torch_dtype=getattr(torch, self.config.torch_dtype)
        )
        
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            torch_dtype=getattr(torch, self.config.torch_dtype)
        )
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            torch_dtype=getattr(torch, self.config.torch_dtype)
        )
        
        # Setup quantization if needed
        quantization_config = None
        if self.config.use_4bit or self.config.use_8bit:
            quantization_config = self._get_quantization_config()
        
        # Load transformer (main trainable component)
        transformer_dtype = getattr(torch, self.config.torch_dtype)
        if quantization_config is not None:
            transformer_dtype = torch.float32
            
        transformer = FluxTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            torch_dtype=transformer_dtype,
            quantization_config=quantization_config,
            device_map=self.config.device_map
        )
        
        # Load scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            revision=self.config.revision,
            cache_dir=self.config.cache_dir
        )
        
        # Store components
        self.model_components = {
            "clip_tokenizer": clip_tokenizer,
            "t5_tokenizer": t5_tokenizer,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "vae": vae,
            "transformer": transformer,
            "scheduler": scheduler
        }
        
        # Freeze non-trainable components
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        
        # Setup LoRA if enabled
        if self.config.use_lora:
            self.setup_lora()
        
        # Set training mode
        transformer.train()
        
        logging.info("FLUX model loaded successfully")
        param_info = get_trainable_parameters_info(transformer)
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
        """Setup LoRA/QLoRA/DoRA for FLUX transformer"""
        try:
            from peft import get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            raise ImportError("PEFT library not available. Install with: pip install peft")
        
        transformer = self.model_components["transformer"]
        
        # Prepare for k-bit training if using quantization
        if self.config.use_4bit or self.config.use_8bit:
            transformer = prepare_model_for_kbit_training(transformer)
        
        # Get target modules
        target_modules = self.config.lora_target_modules or [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2"
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
        transformer = get_peft_model(transformer, lora_config)
        self.model_components["transformer"] = transformer
        
        logging.info(f"Applied {self.config.lora_type.upper()} with rank {self.config.lora_rank}")
        
        # Print trainable parameters
        if hasattr(transformer, 'print_trainable_parameters'):
            transformer.print_trainable_parameters()
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters"""
        return [p for p in self.model_components["transformer"].parameters() if p.requires_grad]
    
    def save_model(self, output_dir: str) -> None:
        """Save FLUX model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        transformer = self.model_components["transformer"]
        
        if self.config.use_lora:
            # Save LoRA weights
            lora_path = output_path / "lora"
            transformer.save_pretrained(lora_path)
            logging.info(f"LoRA weights saved to: {lora_path}")
        else:
            # Save full transformer
            transformer_path = output_path / "transformer"
            transformer.save_pretrained(transformer_path)
            logging.info(f"Full transformer saved to: {transformer_path}")
        
        # Save other components if needed
        for name, component in self.model_components.items():
            if name != "transformer" and hasattr(component, 'save_pretrained'):
                component_path = output_path / name
                component.save_pretrained(component_path)
    
    def load_pretrained(self, model_path: str) -> None:
        """Load pretrained FLUX weights"""
        model_path = Path(model_path)
        
        if self.config.use_lora:
            # Load LoRA weights
            lora_path = model_path / "lora"
            if lora_path.exists():
                try:
                    from peft import PeftModel
                    transformer = self.model_components["transformer"]
                    transformer = PeftModel.from_pretrained(transformer, lora_path)
                    self.model_components["transformer"] = transformer
                    logging.info(f"LoRA weights loaded from: {lora_path}")
                except ImportError:
                    raise ImportError("PEFT library not available for loading LoRA weights")
        else:
            # Load full transformer
            transformer_path = model_path / "transformer"
            if transformer_path.exists():
                transformer = self.model_components["transformer"]
                transformer.load_state_dict(torch.load(transformer_path / "pytorch_model.bin"))
                logging.info(f"Full transformer loaded from: {transformer_path}")
    
    def encode_prompt(self, prompt: str, max_length: int = 512) -> tuple:
        """Encode text prompt using CLIP and T5 encoders"""
        clip_tokenizer = self.model_components["clip_tokenizer"]
        t5_tokenizer = self.model_components["t5_tokenizer"]
        text_encoder = self.model_components["text_encoder"]
        text_encoder_2 = self.model_components["text_encoder_2"]
        
        # CLIP encoding
        clip_inputs = clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            clip_embeddings = text_encoder(
                clip_inputs.input_ids.to(text_encoder.device),
                attention_mask=clip_inputs.attention_mask.to(text_encoder.device)
            )[0]
        
        # T5 encoding
        t5_inputs = t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            t5_embeddings = text_encoder_2(
                t5_inputs.input_ids.to(text_encoder_2.device),
                attention_mask=t5_inputs.attention_mask.to(text_encoder_2.device)
            )[0]
        
        return clip_embeddings, t5_embeddings
