"""
Diffusion Model Finetuning Toolkit
A standardized, normalized pipeline for finetuning FLUX, Wan2.1, and other diffusion models
"""

from .core import (
    BaseTrainer,
    BaseModel,
    BaseDataset,
    TrainingConfig,
    ModelConfig,
    DatasetConfig
)

from .models import (
    FluxModel,
    Wan21ImageToVideoModel,
    Wan21TextToVideoModel
)

from .datasets import (
    TextToImageDataset,
    ImageToVideoDataset,
    TextToVideoDataset,
    AutoCaptionDataset
)

from .trainers import (
    FluxTrainer,
    Wan21I2VTrainer,
    Wan21T2VTrainer
)

from .utils import (
    LoRAConfig,
    QLoRAConfig,
    DoRAConfig,
    setup_logging,
    setup_accelerator,
    save_checkpoint,
    load_checkpoint
)

from .pipeline import (
    TrainingPipeline,
    InferencePipeline,
    DataPreparationPipeline
)

__version__ = "1.0.0"
__author__ = "Diffusion Training Toolkit"

# Registry for models and trainers
MODEL_REGISTRY = {
    "flux": FluxModel,
    "wan2_1_i2v": Wan21ImageToVideoModel,
    "wan2_1_t2v": Wan21TextToVideoModel
}

TRAINER_REGISTRY = {
    "flux": FluxTrainer,
    "wan2_1_i2v": Wan21I2VTrainer,
    "wan2_1_t2v": Wan21T2VTrainer
}

DATASET_REGISTRY = {
    "text_to_image": TextToImageDataset,
    "image_to_video": ImageToVideoDataset,
    "text_to_video": TextToVideoDataset,
    "auto_caption": AutoCaptionDataset
}

def create_model(model_type: str, config: ModelConfig):
    """Factory function to create models"""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](config)

def create_trainer(model_type: str, model, config: TrainingConfig):
    """Factory function to create trainers"""
    if model_type not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown trainer type: {model_type}. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[model_type](model, config)

def create_dataset(dataset_type: str, config: DatasetConfig):
    """Factory function to create datasets"""
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_type](config)
