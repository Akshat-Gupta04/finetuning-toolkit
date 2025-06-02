"""
Main pipeline classes for the diffusion training toolkit
"""

import os
import logging
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path

import torch
from accelerate import Accelerator

from .core import ModelConfig, DatasetConfig, TrainingConfig
from .utils import (
    setup_logging, 
    setup_accelerator, 
    set_random_seed,
    create_output_directory,
    validate_config,
    get_gpu_memory_info,
    format_time
)
from . import create_model, create_trainer, create_dataset


class TrainingPipeline:
    """Main training pipeline for diffusion models"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize training pipeline"""
        # Validate configuration
        validate_config(config)
        
        # Parse configurations
        self.model_config = ModelConfig(**config["model"])
        self.dataset_config = DatasetConfig(**config["dataset"])
        self.training_config = TrainingConfig(**config["training"])
        
        # Store full config
        self.config = config
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.dataset = None
        self.accelerator = None
        
        # Setup logging
        setup_logging(
            self.training_config.logging_dir,
            log_level="INFO"
        )
        
        logging.info("Training pipeline initialized")
        logging.info(f"Model: {self.model_config.model_name}")
        logging.info(f"Dataset: {self.dataset_config.dataset_type}")
        logging.info(f"Output: {self.training_config.output_dir}")
    
    def setup(self) -> None:
        """Setup all pipeline components"""
        start_time = time.time()
        
        # Create output directory
        create_output_directory(self.training_config.output_dir)
        
        # Set random seed
        if self.training_config.seed is not None:
            set_random_seed(self.training_config.seed)
            logging.info(f"Random seed set to: {self.training_config.seed}")
        
        # Setup accelerator
        self.accelerator = setup_accelerator(
            output_dir=self.training_config.output_dir,
            logging_dir=self.training_config.logging_dir,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            mixed_precision=self.training_config.mixed_precision,
            report_to=self.training_config.report_to
        )
        
        # Log GPU info
        gpu_info = get_gpu_memory_info()
        logging.info(f"GPU Memory: {gpu_info['total_mb']:.1f}MB total, {gpu_info['free_mb']:.1f}MB free")
        
        # Create model
        logging.info("Loading model...")
        self.model = create_model(self.model_config.model_name, self.model_config)
        self.model.load_model(self.accelerator)
        
        # Create dataset
        logging.info("Preparing dataset...")
        self.dataset = create_dataset(self.dataset_config.dataset_type, self.dataset_config)
        
        # Create trainer
        logging.info("Setting up trainer...")
        self.trainer = create_trainer(self.model_config.model_name, self.model, self.training_config)
        self.trainer.accelerator = self.accelerator
        self.trainer.setup_training(self.dataset)
        
        setup_time = time.time() - start_time
        logging.info(f"Pipeline setup completed in {format_time(setup_time)}")
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Run training"""
        if self.trainer is None:
            raise RuntimeError("Pipeline not setup. Call setup() first.")
        
        logging.info("Starting training...")
        start_time = time.time()
        
        try:
            self.trainer.train(self.dataset, resume_from_checkpoint)
            
            training_time = time.time() - start_time
            logging.info(f"Training completed in {format_time(training_time)}")
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
    
    def save_final_model(self) -> None:
        """Save the final trained model"""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        final_model_dir = os.path.join(self.training_config.output_dir, "final_model")
        self.model.save_model(final_model_dir)
        logging.info(f"Final model saved to: {final_model_dir}")
    
    def run(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Run the complete training pipeline"""
        try:
            self.setup()
            self.train(resume_from_checkpoint)
            self.save_final_model()
            logging.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            raise


class InferencePipeline:
    """Pipeline for running inference with trained models"""
    
    def __init__(self, model_config: ModelConfig, model_path: str):
        """Initialize inference pipeline"""
        self.model_config = model_config
        self.model_path = model_path
        self.model = None
        self.pipeline = None
        
        setup_logging(log_level="INFO")
        logging.info("Inference pipeline initialized")
    
    def setup(self) -> None:
        """Setup inference pipeline"""
        # Create model
        self.model = create_model(self.model_config.model_name, self.model_config)
        
        # Load model without accelerator for inference
        self.model.load_model(accelerator=None)
        
        # Load trained weights
        self.model.load_pretrained(self.model_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        logging.info("Inference pipeline setup completed")
    
    def generate(self, **kwargs) -> Any:
        """Generate content using the model"""
        if self.model is None:
            raise RuntimeError("Pipeline not setup. Call setup() first.")
        
        # Implementation depends on model type
        model_name = self.model_config.model_name
        
        if model_name == "flux":
            return self._generate_flux(**kwargs)
        elif model_name == "wan2_1_i2v":
            return self._generate_wan21_i2v(**kwargs)
        elif model_name == "wan2_1_t2v":
            return self._generate_wan21_t2v(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def _generate_flux(self, prompt: str, **kwargs) -> Any:
        """Generate image using FLUX"""
        try:
            from diffusers import FluxPipeline
        except ImportError:
            raise ImportError("diffusers library required for FLUX inference")
        
        # Create pipeline
        pipeline = FluxPipeline(
            vae=self.model.model_components["vae"],
            text_encoder=self.model.model_components["text_encoder"],
            text_encoder_2=self.model.model_components["text_encoder_2"],
            tokenizer=self.model.model_components["clip_tokenizer"],
            tokenizer_2=self.model.model_components["t5_tokenizer"],
            transformer=self.model.model_components["transformer"],
            scheduler=self.model.model_components["scheduler"]
        )
        
        # Generate
        with torch.no_grad():
            result = pipeline(prompt=prompt, **kwargs)
        
        return result.images[0]
    
    def _generate_wan21_i2v(self, image, prompt: str, **kwargs) -> Any:
        """Generate video using Wan2.1 I2V"""
        # Implementation for Wan2.1 I2V inference
        # This would use the specific Wan2.1 I2V pipeline
        raise NotImplementedError("Wan2.1 I2V inference not yet implemented")
    
    def _generate_wan21_t2v(self, prompt: str, **kwargs) -> Any:
        """Generate video using Wan2.1 T2V"""
        # Implementation for Wan2.1 T2V inference
        # This would use the specific Wan2.1 T2V pipeline
        raise NotImplementedError("Wan2.1 T2V inference not yet implemented")


class DataPreparationPipeline:
    """Pipeline for preparing datasets with auto-captioning"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data preparation pipeline"""
        self.config = config
        self.dataset_config = DatasetConfig(**config["dataset"])
        
        setup_logging(log_level="INFO")
        logging.info("Data preparation pipeline initialized")
    
    def prepare_dataset(self) -> None:
        """Prepare dataset with auto-captioning if needed"""
        if not self.dataset_config.auto_caption:
            logging.info("Auto-captioning disabled, skipping data preparation")
            return
        
        logging.info("Starting dataset preparation with auto-captioning...")
        
        try:
            from ..src.dataset_auto import AutoDatasetProcessor
        except ImportError:
            logging.error("Auto dataset processor not available")
            return
        
        # Create processor
        processor = AutoDatasetProcessor(
            input_dir=self.dataset_config.data_dir,
            output_dir=self.dataset_config.data_dir,
            model_type=self.dataset_config.dataset_type,
            captioning_model=self.dataset_config.captioning_model,
            quality_threshold=self.dataset_config.quality_threshold
        )
        
        # Process dataset
        results = processor.process_dataset()
        
        logging.info(f"Dataset preparation completed: {results}")
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate prepared dataset"""
        data_dir = Path(self.dataset_config.data_dir)
        
        if not data_dir.exists():
            return {"valid": False, "error": "Data directory does not exist"}
        
        # Count files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        image_count = 0
        video_count = 0
        caption_count = 0
        
        for file_path in data_dir.rglob("*"):
            if file_path.suffix.lower() in image_extensions:
                image_count += 1
            elif file_path.suffix.lower() in video_extensions:
                video_count += 1
            elif file_path.suffix == '.txt':
                caption_count += 1
        
        # Check for metadata
        metadata_files = list(data_dir.glob("metadata.*"))
        
        return {
            "valid": True,
            "image_count": image_count,
            "video_count": video_count,
            "caption_count": caption_count,
            "metadata_files": [str(f) for f in metadata_files],
            "data_dir": str(data_dir)
        }


def create_training_pipeline(config_path: str) -> TrainingPipeline:
    """Factory function to create training pipeline from config file"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return TrainingPipeline(config)


def create_inference_pipeline(model_type: str, model_path: str, **model_kwargs) -> InferencePipeline:
    """Factory function to create inference pipeline"""
    model_config = ModelConfig(
        model_name=model_type,
        pretrained_model_path=model_path,
        **model_kwargs
    )
    
    return InferencePipeline(model_config, model_path)
