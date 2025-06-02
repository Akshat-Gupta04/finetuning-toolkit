"""
Core base classes and configurations for the diffusion training toolkit
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator


@dataclass
class ModelConfig:
    """Configuration for model loading and setup"""
    model_name: str
    pretrained_model_path: str
    cache_dir: Optional[str] = "./cache"
    revision: Optional[str] = None
    torch_dtype: str = "bfloat16"
    device_map: Optional[str] = None

    # LoRA/QLoRA/DoRA configuration
    use_lora: bool = True
    lora_type: str = "lora"  # lora, qlora, dora
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None

    # Quantization
    use_4bit: bool = False
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Model-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing"""
    dataset_type: str
    data_dir: str
    image_column: str = "image"
    caption_column: str = "caption"
    video_column: str = "video"

    # Processing parameters
    resolution: Union[int, tuple] = 512
    max_resolution: int = 2048
    min_resolution: int = 256
    center_crop: bool = True
    random_flip: float = 0.0

    # Text processing
    max_sequence_length: int = 512

    # Auto-captioning
    auto_caption: bool = False
    captioning_model: str = "blip2"
    quality_threshold: float = 0.7

    # Video-specific
    num_frames: int = 16
    frame_rate: int = 8

    # Variable size training
    variable_size: bool = True
    bucket_sizes: Optional[List[tuple]] = None

    # Extra parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    output_dir: str
    logging_dir: Optional[str] = None

    # Training parameters
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1

    # Optimization
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Mixed precision and acceleration
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True

    # Logging and validation
    logging_steps: int = 10
    validation_steps: int = 500
    checkpointing_steps: int = 500

    # Reproducibility
    seed: Optional[int] = 42

    # Monitoring
    report_to: Optional[str] = None  # wandb, tensorboard, etc.

    # Hardware optimization
    dataloader_num_workers: int = 4
    pin_memory: bool = True

    # Extra parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Base class for all diffusion models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_components = {}
        self.device = None

    @abstractmethod
    def load_model(self, accelerator: Accelerator) -> Dict[str, Any]:
        """Load model components"""
        pass

    @abstractmethod
    def setup_lora(self) -> None:
        """Setup LoRA/QLoRA/DoRA if enabled"""
        pass

    @abstractmethod
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get parameters that should be trained"""
        pass

    @abstractmethod
    def save_model(self, output_dir: str) -> None:
        """Save model to directory"""
        pass

    @abstractmethod
    def load_pretrained(self, model_path: str) -> None:
        """Load pretrained weights"""
        pass

    def to(self, device):
        """Move model to device"""
        self.device = device
        for component in self.model_components.values():
            if hasattr(component, 'to'):
                component.to(device)
        return self

    def train(self):
        """Set model to training mode"""
        for component in self.model_components.values():
            if hasattr(component, 'train'):
                component.train()

    def eval(self):
        """Set model to evaluation mode"""
        for component in self.model_components.values():
            if hasattr(component, 'eval'):
                component.eval()


class BaseDataset(ABC, Dataset):
    """Base class for all datasets"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data = []
        self.prepare_data()

    @abstractmethod
    def prepare_data(self) -> None:
        """Prepare dataset from configuration"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index"""
        pass

    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.data)

    @abstractmethod
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for DataLoader"""
        pass


class BaseTrainer(ABC):
    """Base class for all trainers"""

    def __init__(self, model: BaseModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.accelerator = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.global_step = 0
        self.epoch = 0

    @abstractmethod
    def setup_training(self, dataset: BaseDataset) -> None:
        """Setup training components"""
        pass

    @abstractmethod
    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute training loss"""
        pass

    @abstractmethod
    def validation_step(self) -> Dict[str, Any]:
        """Run validation step"""
        pass

    def setup_accelerator(self) -> Accelerator:
        """Setup Accelerate for distributed training"""
        from accelerate.utils import ProjectConfiguration

        project_config = ProjectConfiguration(
            project_dir=self.config.output_dir,
            logging_dir=self.config.logging_dir or os.path.join(self.config.output_dir, "logs")
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with=self.config.report_to,
            project_config=project_config
        )

        return self.accelerator

    def setup_optimizer(self) -> None:
        """Setup optimizer"""
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                logging.warning("bitsandbytes not available, using regular AdamW")
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            self.model.get_trainable_parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon
        )

    def setup_scheduler(self) -> None:
        """Setup learning rate scheduler"""
        from diffusers.optimization import get_scheduler

        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=self.config.max_train_steps,
            num_cycles=self.config.lr_num_cycles
        )

    def save_checkpoint(self, step: int) -> None:
        """Save training checkpoint"""
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
            self.accelerator.save_state(save_path)
            logging.info(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.global_step = int(checkpoint_path.split("-")[-1])
        logging.info(f"Loaded checkpoint: {checkpoint_path}")

    @abstractmethod
    def train(self, dataset: BaseDataset, resume_from_checkpoint: Optional[str] = None) -> None:
        """Main training loop"""
        pass

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training metrics"""
        if self.config.report_to and self.accelerator.is_main_process:
            if self.config.report_to == "wandb":
                try:
                    import wandb
                    wandb.log(metrics, step=step)
                except ImportError:
                    logging.warning("wandb not available for logging")
            elif self.config.report_to == "tensorboard":
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    writer = SummaryWriter(self.config.logging_dir)
                    for key, value in metrics.items():
                        writer.add_scalar(key, value, step)
                except ImportError:
                    logging.warning("tensorboard not available for logging")
