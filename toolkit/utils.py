"""
Utility functions and configurations for the diffusion training toolkit
"""

import os
import logging
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

import torch
import numpy as np
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed


@dataclass
class LoRAConfig:
    """Configuration for LoRA training"""
    rank: int = 64
    alpha: int = 64
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "DIFFUSION"
    
    def to_peft_config(self):
        """Convert to PEFT LoraConfig"""
        try:
            from peft import LoraConfig as PeftLoraConfig
            return PeftLoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                target_modules=self.target_modules,
                lora_dropout=self.dropout,
                bias=self.bias,
                task_type=self.task_type
            )
        except ImportError:
            raise ImportError("PEFT library not available. Install with: pip install peft")


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (4-bit quantized LoRA) training"""
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    def to_quantization_config(self):
        """Convert to BitsAndBytesConfig"""
        try:
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.bnb_4bit_compute_dtype)
            )
        except ImportError:
            raise ImportError("transformers library not available or too old")


@dataclass
class DoRAConfig(LoRAConfig):
    """Configuration for DoRA (Weight-Decomposed LoRA) training"""
    use_dora: bool = True
    
    def to_peft_config(self):
        """Convert to PEFT LoraConfig with DoRA enabled"""
        try:
            from peft import LoraConfig as PeftLoraConfig
            return PeftLoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                target_modules=self.target_modules,
                lora_dropout=self.dropout,
                bias=self.bias,
                task_type=self.task_type,
                use_dora=self.use_dora
            )
        except ImportError:
            raise ImportError("PEFT library not available. Install with: pip install peft")


def setup_logging(
    logging_dir: Optional[str] = None,
    log_level: str = "INFO",
    accelerator: Optional[Accelerator] = None
) -> None:
    """Setup logging configuration"""
    log_level = getattr(logging, log_level.upper())
    
    if accelerator is None or accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=log_level,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(logging_dir or ".", "training.log"))
            ] if logging_dir else [logging.StreamHandler()]
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def setup_accelerator(
    output_dir: str,
    logging_dir: Optional[str] = None,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "bf16",
    report_to: Optional[str] = None
) -> Accelerator:
    """Setup Accelerate for distributed training"""
    project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=logging_dir or os.path.join(output_dir, "logs")
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_config=project_config
    )
    
    return accelerator


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def save_checkpoint(
    accelerator: Accelerator,
    output_dir: str,
    step: int,
    extra_data: Optional[Dict[str, Any]] = None
) -> str:
    """Save training checkpoint"""
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    accelerator.save_state(save_path)
    
    if extra_data and accelerator.is_main_process:
        import json
        with open(os.path.join(save_path, "extra_data.json"), "w") as f:
            json.dump(extra_data, f, indent=2)
    
    return save_path


def load_checkpoint(
    accelerator: Accelerator,
    checkpoint_path: str
) -> Dict[str, Any]:
    """Load training checkpoint"""
    accelerator.load_state(checkpoint_path)
    
    extra_data = {}
    extra_data_path = os.path.join(checkpoint_path, "extra_data.json")
    if os.path.exists(extra_data_path):
        import json
        with open(extra_data_path, "r") as f:
            extra_data = json.load(f)
    
    return extra_data


def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """Get model memory usage statistics"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        "param_size_mb": param_size / 1024 / 1024,
        "buffer_size_mb": buffer_size / 1024 / 1024,
        "total_size_mb": total_size / 1024 / 1024
    }


def get_trainable_parameters_info(model: torch.nn.Module) -> Dict[str, Any]:
    """Get information about trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
        "frozen_parameters": total_params - trainable_params
    }


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def create_output_directory(output_dir: str, exist_ok: bool = True) -> Path:
    """Create output directory with proper structure"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=exist_ok)
    
    # Create subdirectories
    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "validation").mkdir(exist_ok=True)
    (output_path / "final_model").mkdir(exist_ok=True)
    
    return output_path


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters"""
    required_keys = ["model", "dataset", "training"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    # Validate model config
    model_config = config["model"]
    if "model_name" not in model_config:
        raise ValueError("model.model_name is required")
    if "pretrained_model_path" not in model_config:
        raise ValueError("model.pretrained_model_path is required")
    
    # Validate dataset config
    dataset_config = config["dataset"]
    if "data_dir" not in dataset_config:
        raise ValueError("dataset.data_dir is required")
    
    # Validate training config
    training_config = config["training"]
    if "output_dir" not in training_config:
        raise ValueError("training.output_dir is required")
    
    # Validate LoRA config if enabled
    if model_config.get("use_lora", False):
        lora_rank = model_config.get("lora_rank", 64)
        if not isinstance(lora_rank, int) or lora_rank <= 0:
            raise ValueError("lora_rank must be a positive integer")
        
        lora_alpha = model_config.get("lora_alpha", 64)
        if not isinstance(lora_alpha, int) or lora_alpha <= 0:
            raise ValueError("lora_alpha must be a positive integer")


def get_optimal_batch_size(
    model_memory_mb: float,
    available_memory_mb: float,
    base_batch_size: int = 1,
    safety_factor: float = 0.8
) -> int:
    """Estimate optimal batch size based on memory constraints"""
    usable_memory = available_memory_mb * safety_factor
    memory_per_sample = model_memory_mb / base_batch_size
    
    optimal_batch_size = int(usable_memory / memory_per_sample)
    return max(1, optimal_batch_size)


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return {"total_mb": 0, "allocated_mb": 0, "free_mb": 0}
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    allocated_memory = torch.cuda.memory_allocated(0) / 1024 / 1024
    free_memory = total_memory - allocated_memory
    
    return {
        "total_mb": total_memory,
        "allocated_mb": allocated_memory,
        "free_mb": free_memory
    }
