"""
LoRA/QLoRA/DoRA utilities for diffusion model training
Supports FLUX, Wan2.1, and Text-to-Video models
"""

import torch
from typing import Dict, Any, Optional, List
import logging

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from peft import prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not available. LoRA training will be disabled.")

try:
    import bitsandbytes
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    logging.warning("bitsandbytes not available. QLoRA training will be disabled.")


def get_lora_target_modules(model_type: str, config: Dict[str, Any]) -> List[str]:
    """Get LoRA target modules for specific model type"""
    lora_target_modules = config["model"].get("lora_target_modules", {})

    if model_type in lora_target_modules:
        return lora_target_modules[model_type]

    # Default target modules for different model types
    default_modules = {
        "flux": ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"],
        "wan2_1_i2v": ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"],
        "wan2_1_t2v": ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
    }

    return default_modules.get(model_type, default_modules["flux"])


def create_lora_config(config: Dict[str, Any], model_type: str) -> Optional[LoraConfig]:
    """Create LoRA configuration based on config and model type"""
    if not PEFT_AVAILABLE:
        logging.error("PEFT not available. Cannot create LoRA config.")
        return None

    model_config = config["model"]
    lora_type = model_config.get("lora_type", "lora").lower()

    # Get LoRA configuration from config
    if "lora_configs" in config and lora_type in config["lora_configs"]:
        lora_config_dict = config["lora_configs"][lora_type]
    else:
        # Fallback to model config
        lora_config_dict = {
            "rank": model_config.get("lora_rank", 64),
            "alpha": model_config.get("lora_alpha", 64),
            "dropout": model_config.get("lora_dropout", 0.1),
            "bias": "none",
            "task_type": "DIFFUSION"
        }

    # Get target modules
    target_modules = get_lora_target_modules(model_type, config)

    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_config_dict.get("rank", 64),
        lora_alpha=lora_config_dict.get("alpha", 64),
        target_modules=target_modules,
        lora_dropout=lora_config_dict.get("dropout", 0.1),
        bias=lora_config_dict.get("bias", "none"),
        task_type=lora_config_dict.get("task_type", "DIFFUSION"),
        use_dora=lora_config_dict.get("use_dora", False)
    )

    return lora_config


def prepare_model_for_lora(model: torch.nn.Module, config: Dict[str, Any], model_type: str) -> torch.nn.Module:
    """Prepare model for LoRA training"""
    if not PEFT_AVAILABLE:
        logging.error("PEFT not available. Returning original model.")
        return model

    model_config = config["model"]

    # Check if LoRA is enabled
    if not model_config.get("use_lora", False):
        return model

    lora_type = model_config.get("lora_type", "lora").lower()

    # Handle QLoRA (4-bit quantization)
    if lora_type == "qlora" or model_config.get("use_qlora", False):
        if not BNB_AVAILABLE:
            logging.error("bitsandbytes not available. Cannot use QLoRA.")
            return model

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        logging.info("Model prepared for QLoRA (4-bit) training")

    # Create LoRA config
    lora_config = create_lora_config(config, model_type)
    if lora_config is None:
        return model

    # Apply LoRA to model
    try:
        model = get_peft_model(model, lora_config)
        logging.info(f"LoRA applied to model with config: {lora_config}")

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    except Exception as e:
        logging.error(f"Failed to apply LoRA to model: {e}")
        return model

    return model


def load_quantized_model(model_path: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load model with quantization for QLoRA"""
    if not BNB_AVAILABLE:
        logging.error("bitsandbytes not available. Cannot load quantized model.")
        return None

    model_config = config["model"]

    # QLoRA quantization config
    if model_config.get("use_qlora", False) or model_config.get("lora_type") == "qlora":
        qlora_config = model_config.get("qlora_config", {})

        quantization_config = {
            "load_in_4bit": qlora_config.get("load_in_4bit", True),
            "bnb_4bit_use_double_quant": qlora_config.get("bnb_4bit_use_double_quant", True),
            "bnb_4bit_quant_type": qlora_config.get("bnb_4bit_quant_type", "nf4"),
            "bnb_4bit_compute_dtype": getattr(torch, qlora_config.get("bnb_4bit_compute_dtype", "bfloat16"))
        }

        logging.info(f"Loading model with QLoRA quantization: {quantization_config}")
        return quantization_config

    # 8-bit quantization
    elif model_config.get("use_8bit", False):
        return {"load_in_8bit": True}

    # 4-bit quantization
    elif model_config.get("use_4bit", False):
        return {"load_in_4bit": True}

    return None


def save_lora_model(model: torch.nn.Module, output_dir: str) -> None:
    """Save LoRA model"""
    if not PEFT_AVAILABLE:
        logging.error("PEFT not available. Cannot save LoRA model.")
        return

    try:
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(output_dir)
            logging.info(f"LoRA model saved to {output_dir}")
        else:
            logging.error("Model does not have save_pretrained method")
    except Exception as e:
        logging.error(f"Failed to save LoRA model: {e}")


def load_lora_model(base_model: torch.nn.Module, lora_path: str) -> torch.nn.Module:
    """Load LoRA weights into base model"""
    if not PEFT_AVAILABLE:
        logging.error("PEFT not available. Cannot load LoRA model.")
        return base_model

    try:
        model = PeftModel.from_pretrained(base_model, lora_path)
        logging.info(f"LoRA weights loaded from {lora_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load LoRA model: {e}")
        return base_model


def get_lora_config_summary(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Get summary of LoRA configuration"""
    model_config = config["model"]

    if not model_config.get("use_lora", False):
        return {"enabled": False}

    lora_type = model_config.get("lora_type", "lora").lower()
    target_modules = get_lora_target_modules(model_type, config)

    summary = {
        "enabled": True,
        "type": lora_type,
        "rank": model_config.get("lora_rank", 64),
        "alpha": model_config.get("lora_alpha", 64),
        "dropout": model_config.get("lora_dropout", 0.1),
        "target_modules": target_modules,
        "quantization": None
    }

    # Add quantization info
    if lora_type == "qlora" or model_config.get("use_qlora", False):
        summary["quantization"] = "4-bit"
    elif model_config.get("use_8bit", False):
        summary["quantization"] = "8-bit"
    elif model_config.get("use_4bit", False):
        summary["quantization"] = "4-bit"

    # Add DoRA info
    if lora_type == "dora" or model_config.get("use_dora", False):
        summary["dora"] = True

    return summary


def validate_lora_config(config: Dict[str, Any], model_type: str) -> bool:
    """Validate LoRA configuration"""
    model_config = config["model"]

    if not model_config.get("use_lora", False):
        return True

    # Check if PEFT is available
    if not PEFT_AVAILABLE:
        logging.error("PEFT not available but LoRA is enabled")
        return False

    # Check QLoRA requirements
    lora_type = model_config.get("lora_type", "lora").lower()
    if lora_type == "qlora" or model_config.get("use_qlora", False):
        if not BNB_AVAILABLE:
            logging.error("bitsandbytes not available but QLoRA is enabled")
            return False

    # Validate rank and alpha
    rank = model_config.get("lora_rank", 64)
    alpha = model_config.get("lora_alpha", 64)

    if rank <= 0 or rank > 1024:
        logging.error(f"Invalid LoRA rank: {rank}. Must be between 1 and 1024.")
        return False

    if alpha <= 0:
        logging.error(f"Invalid LoRA alpha: {alpha}. Must be positive.")
        return False

    # Validate target modules
    target_modules = get_lora_target_modules(model_type, config)
    if not target_modules:
        logging.error(f"No target modules specified for model type: {model_type}")
        return False

    logging.info("LoRA configuration validation passed")
    return True
