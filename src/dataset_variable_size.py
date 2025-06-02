"""
Enhanced dataset utilities for variable-sized images without cropping
Supports both Wan2.1 and FLUX models with proper aspect ratio preservation
"""

import os
import json
import random
import math
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTokenizer, T5TokenizerFast
import decord
from decord import VideoReader, cpu
import imageio

decord.bridge.set_bridge('torch')


class VariableSizeImageVideoDataset(Dataset):
    """
    Enhanced dataset for Wan2.1 that handles variable-sized images without cropping
    Preserves aspect ratios and uses intelligent padding/resizing
    """
    
    def __init__(
        self,
        data_dir: str,
        image_column: str = "image",
        video_column: str = "video", 
        caption_column: str = "caption",
        max_width: int = 1280,
        max_height: int = 720,
        num_frames: int = 81,
        frame_rate: int = 16,
        sample_stride: int = 1,
        random_flip: float = 0.0,
        normalize: bool = True,
        max_sequence_length: int = 256,
        resize_mode: str = "pad",  # "pad", "crop", "stretch"
        bucket_sizes: Optional[List[Tuple[int, int]]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.image_column = image_column
        self.video_column = video_column
        self.caption_column = caption_column
        
        self.max_width = max_width
        self.max_height = max_height
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.sample_stride = sample_stride
        self.random_flip = random_flip
        self.normalize = normalize
        self.max_sequence_length = max_sequence_length
        self.resize_mode = resize_mode
        
        # Aspect ratio buckets for efficient batching
        if bucket_sizes is None:
            self.bucket_sizes = [
                (512, 512), (576, 576), (640, 640), (704, 704), (768, 768),
                (832, 832), (896, 896), (960, 960), (1024, 1024),
                (512, 768), (576, 864), (640, 960), (704, 1056),
                (768, 512), (864, 576), (960, 640), (1056, 704),
                (1280, 720), (1024, 576), (960, 540), (854, 480)
            ]
        else:
            self.bucket_sizes = bucket_sizes
            
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Analyze dataset and assign buckets
        self.analyze_dataset()
        
        # Setup transforms
        self.image_processor = CLIPImageProcessor()
        self.setup_transforms()
        
    def analyze_dataset(self):
        """Analyze dataset to determine optimal bucket assignments"""
        print("Analyzing dataset for optimal aspect ratio bucketing...")
        
        aspect_ratios = []
        for sample in self.metadata:
            image_path = self.data_dir / sample[self.image_column]
            if image_path.exists():
                try:
                    with Image.open(image_path) as img:
                        aspect_ratios.append(img.width / img.height)
                except:
                    aspect_ratios.append(16/9)  # Default aspect ratio
            else:
                aspect_ratios.append(16/9)
                
        # Assign each sample to the best bucket
        self.sample_buckets = []
        for i, sample in enumerate(self.metadata):
            if i < len(aspect_ratios):
                target_ratio = aspect_ratios[i]
                best_bucket = self.find_best_bucket(target_ratio)
                self.sample_buckets.append(best_bucket)
            else:
                self.sample_buckets.append((self.max_width, self.max_height))
                
        print(f"Dataset analyzed: {len(self.metadata)} samples assigned to buckets")
        
    def find_best_bucket(self, target_ratio: float) -> Tuple[int, int]:
        """Find the best bucket size for a given aspect ratio"""
        best_bucket = self.bucket_sizes[0]
        best_diff = float('inf')
        
        for width, height in self.bucket_sizes:
            bucket_ratio = width / height
            diff = abs(bucket_ratio - target_ratio)
            
            if diff < best_diff:
                best_diff = diff
                best_bucket = (width, height)
                
        return best_bucket
        
    def setup_transforms(self):
        """Setup basic transforms without resizing"""
        transform_list = [transforms.ToTensor()]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
            
        self.basic_transform = transforms.Compose(transform_list)
        
    def resize_with_aspect_ratio(self, image, target_width, target_height, mode="pad"):
        """
        Resize image while preserving aspect ratio
        
        Args:
            image: PIL Image
            target_width: Target width
            target_height: Target height
            mode: "pad", "crop", or "stretch"
            
        Returns:
            PIL Image resized to target dimensions
        """
        if mode == "stretch":
            # Simple stretch - may distort image
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
        elif mode == "crop":
            # Smart crop to target aspect ratio
            return self.smart_crop(image, target_width, target_height)
            
        else:  # mode == "pad"
            # Pad to maintain aspect ratio
            return self.pad_to_size(image, target_width, target_height)
            
    def smart_crop(self, image, target_width, target_height):
        """Smart crop that preserves important content"""
        original_width, original_height = image.size
        target_ratio = target_width / target_height
        original_ratio = original_width / original_height
        
        if abs(target_ratio - original_ratio) < 0.01:
            # Aspect ratios are very close, just resize
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
        if original_ratio > target_ratio:
            # Image is wider, crop width
            new_width = int(original_height * target_ratio)
            left = (original_width - new_width) // 2
            crop_box = (left, 0, left + new_width, original_height)
        else:
            # Image is taller, crop height
            new_height = int(original_width / target_ratio)
            top = (original_height - new_height) // 2
            crop_box = (0, top, original_width, top + new_height)
            
        cropped = image.crop(crop_box)
        return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
    def pad_to_size(self, image, target_width, target_height, pad_color=(0, 0, 0)):
        """Pad image to target size while maintaining aspect ratio"""
        original_width, original_height = image.size
        
        # Calculate scaling factor
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create padded image
        padded_image = Image.new('RGB', (target_width, target_height), pad_color)
        
        # Calculate padding offsets
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste resized image
        padded_image.paste(resized_image, (paste_x, paste_y))
        
        return padded_image
        
    def __len__(self):
        return len(self.metadata)
        
    def load_image(self, image_path: str, target_width: int, target_height: int) -> Image.Image:
        """Load and preprocess image with variable size support"""
        full_path = self.data_dir / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
            
        image = Image.open(full_path).convert('RGB')
        
        # Apply random flip
        if random.random() < self.random_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
        # Resize to target dimensions
        image = self.resize_with_aspect_ratio(image, target_width, target_height, self.resize_mode)
        
        return image
        
    def load_video(self, video_path: str, target_width: int, target_height: int) -> torch.Tensor:
        """Load and preprocess video with variable size support"""
        full_path = self.data_dir / video_path
        if not full_path.exists():
            raise FileNotFoundError(f"Video not found: {full_path}")
            
        try:
            # Use decord for efficient video loading
            vr = VideoReader(str(full_path), ctx=cpu(0))
            total_frames = len(vr)
            
            # Sample frames
            if total_frames < self.num_frames:
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                start_idx = random.randint(0, max(0, total_frames - self.num_frames * self.sample_stride))
                frame_indices = np.arange(start_idx, start_idx + self.num_frames * self.sample_stride, self.sample_stride)
                frame_indices = frame_indices[:self.num_frames]
                
            # Load frames
            frames = vr.get_batch(frame_indices).asnumpy()
            
            # Process frames with variable size support
            video_frames = []
            for frame in frames:
                frame_pil = Image.fromarray(frame)
                
                # Apply random flip
                if random.random() < self.random_flip:
                    frame_pil = frame_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    
                # Resize to target dimensions
                frame_pil = self.resize_with_aspect_ratio(frame_pil, target_width, target_height, self.resize_mode)
                
                frame_tensor = self.basic_transform(frame_pil)
                video_frames.append(frame_tensor)
                
            video_tensor = torch.stack(video_frames, dim=0)
            
        except Exception as e:
            print(f"Error loading video {full_path}: {e}")
            # Fallback: create dummy video
            video_tensor = torch.zeros(self.num_frames, 3, target_height, target_width)
            
        return video_tensor
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with variable size support"""
        sample = self.metadata[idx]
        
        # Get target dimensions for this sample
        target_width, target_height = self.sample_buckets[idx]
        
        # Load image
        image_path = sample[self.image_column]
        image = self.load_image(image_path, target_width, target_height)
        
        # Process image for CLIP encoder (fixed size)
        image_clip = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        
        # Transform image for VAE
        image_vae = self.basic_transform(image)
        
        # Load video
        video_path = sample[self.video_column]
        video = self.load_video(video_path, target_width, target_height)
        
        # Get caption
        caption = sample.get(self.caption_column, "")
        if len(caption) > self.max_sequence_length:
            caption = caption[:self.max_sequence_length]
            
        return {
            "image_clip": image_clip,
            "image_vae": image_vae,
            "video": video,
            "caption": caption,
            "image_path": image_path,
            "video_path": video_path,
            "target_width": target_width,
            "target_height": target_height,
            "bucket_id": f"{target_width}x{target_height}",
        }


class VariableSizeFluxDataset(Dataset):
    """
    Enhanced FLUX dataset that handles variable-sized images without cropping
    """
    
    def __init__(
        self,
        data_dir: str,
        image_column: str = "image",
        caption_column: str = "caption",
        max_resolution: int = 1024,
        random_flip: float = 0.0,
        normalize: bool = True,
        max_sequence_length: int = 512,
        resize_mode: str = "pad",  # "pad", "crop", "stretch"
        bucket_sizes: Optional[List[int]] = None,
        tokenizer_name: str = "openai/clip-vit-large-patch14",
        t5_tokenizer_name: str = "google/t5-v1_1-xxl",
    ):
        self.data_dir = Path(data_dir)
        self.image_column = image_column
        self.caption_column = caption_column
        
        self.max_resolution = max_resolution
        self.random_flip = random_flip
        self.normalize = normalize
        self.max_sequence_length = max_sequence_length
        self.resize_mode = resize_mode
        
        # Square bucket sizes for FLUX
        if bucket_sizes is None:
            self.bucket_sizes = [512, 576, 640, 704, 768, 832, 896, 960, 1024]
        else:
            self.bucket_sizes = bucket_sizes
            
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Setup tokenizers
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_tokenizer_name)
        
        # Analyze dataset and assign buckets
        self.analyze_dataset()
        
        # Setup transforms
        self.setup_transforms()
        
    def analyze_dataset(self):
        """Analyze dataset to determine optimal bucket assignments"""
        print("Analyzing FLUX dataset for optimal size bucketing...")
        
        image_sizes = []
        for sample in self.metadata:
            image_path = self.data_dir / sample[self.image_column]
            if image_path.exists():
                try:
                    with Image.open(image_path) as img:
                        # For FLUX, we use the larger dimension
                        max_dim = max(img.width, img.height)
                        image_sizes.append(max_dim)
                except:
                    image_sizes.append(1024)  # Default size
            else:
                image_sizes.append(1024)
                
        # Assign each sample to the best bucket
        self.sample_buckets = []
        for size in image_sizes:
            best_bucket = min(self.bucket_sizes, key=lambda x: abs(x - size))
            self.sample_buckets.append(best_bucket)
            
        print(f"FLUX dataset analyzed: {len(self.metadata)} samples assigned to buckets")
        
    def setup_transforms(self):
        """Setup basic transforms without resizing"""
        transform_list = [transforms.ToTensor()]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
            
        self.basic_transform = transforms.Compose(transform_list)
        
    def resize_with_aspect_ratio(self, image, target_size, mode="pad"):
        """Resize image to square target size while preserving aspect ratio"""
        if mode == "stretch":
            return image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        elif mode == "crop":
            return self.smart_crop_square(image, target_size)
        else:  # mode == "pad"
            return self.pad_to_square(image, target_size)
            
    def smart_crop_square(self, image, target_size):
        """Smart crop to square"""
        width, height = image.size
        
        if width == height:
            return image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
        # Crop to square
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        
        cropped = image.crop((left, top, left + crop_size, top + crop_size))
        return cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
    def pad_to_square(self, image, target_size, pad_color=(0, 0, 0)):
        """Pad image to square while maintaining aspect ratio"""
        width, height = image.size
        
        # Calculate scaling factor
        scale = target_size / max(width, height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create square padded image
        padded_image = Image.new('RGB', (target_size, target_size), pad_color)
        
        # Calculate padding offsets
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        
        # Paste resized image
        padded_image.paste(resized_image, (paste_x, paste_y))
        
        return padded_image
        
    def tokenize_caption(self, caption: str) -> Dict[str, torch.Tensor]:
        """Tokenize caption for both CLIP and T5 tokenizers"""
        # CLIP tokenization
        clip_inputs = self.clip_tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        # T5 tokenization
        t5_inputs = self.t5_tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "clip_input_ids": clip_inputs.input_ids.squeeze(0),
            "clip_attention_mask": clip_inputs.attention_mask.squeeze(0),
            "t5_input_ids": t5_inputs.input_ids.squeeze(0),
            "t5_attention_mask": t5_inputs.attention_mask.squeeze(0),
        }
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with variable size support"""
        sample = self.metadata[idx]
        
        # Get target size for this sample
        target_size = self.sample_buckets[idx]
        
        # Load image
        image_path = sample[self.image_column]
        full_path = self.data_dir / image_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
            
        image = Image.open(full_path).convert('RGB')
        
        # Apply random flip
        if random.random() < self.random_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
        # Resize to target size
        image = self.resize_with_aspect_ratio(image, target_size, self.resize_mode)
        
        # Transform image
        image_tensor = self.basic_transform(image)
        
        # Get caption
        caption = sample.get(self.caption_column, "")
        if len(caption) > self.max_sequence_length:
            caption = caption[:self.max_sequence_length]
            
        # Tokenize caption
        tokenized = self.tokenize_caption(caption)
        
        return {
            "pixel_values": image_tensor,
            "input_ids_clip": tokenized["clip_input_ids"],
            "attention_mask_clip": tokenized["clip_attention_mask"],
            "input_ids_t5": tokenized["t5_input_ids"],
            "attention_mask_t5": tokenized["t5_attention_mask"],
            "caption": caption,
            "image_path": image_path,
            "target_size": target_size,
            "bucket_id": f"{target_size}x{target_size}",
        }
