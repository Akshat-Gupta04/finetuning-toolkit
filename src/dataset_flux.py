"""
Dataset utilities for FLUX.1-dev text-to-image finetuning
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, T5TokenizerFast


class FluxTextToImageDataset(Dataset):
    """
    Dataset for FLUX.1-dev text-to-image training

    Expected data structure:
    data_dir/
    ├── metadata.json  # Contains image_path, caption for each sample
    └── images/
        ├── image1.jpg
        └── image2.jpg
    """

    def __init__(
        self,
        data_dir: str,
        image_column: str = "image",
        caption_column: str = "caption",
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: float = 0.0,
        normalize: bool = True,
        max_sequence_length: int = 512,
        tokenizer_name: str = "openai/clip-vit-large-patch14",
        t5_tokenizer_name: str = "google/t5-v1_1-xxl",
    ):
        self.data_dir = Path(data_dir)
        self.image_column = image_column
        self.caption_column = caption_column

        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.normalize = normalize
        self.max_sequence_length = max_sequence_length

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Setup tokenizers for FLUX (uses both CLIP and T5)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_tokenizer_name)

        # Setup transforms
        self.setup_transforms()

    def setup_transforms(self):
        """Setup image transforms with aspect ratio preservation"""
        # We'll handle resizing manually to preserve aspect ratios
        # Only basic transforms here
        transform_list = [transforms.ToTensor()]

        # Normalize to [-1, 1] for diffusion models
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )

        self.image_transform = transforms.Compose(transform_list)

    def resize_with_aspect_ratio(self, image, target_size, pad_color=(0, 0, 0)):
        """
        Resize image while maintaining aspect ratio and pad to square

        Args:
            image: PIL Image
            target_size: Target size (width and height for square)
            pad_color: Color for padding (RGB tuple)

        Returns:
            PIL Image resized and padded to target size
        """
        original_width, original_height = image.size

        # Calculate scaling factor to fit within target dimensions
        scale = min(target_size / original_width, target_size / original_height)

        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image maintaining aspect ratio
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new square image with target size and pad color
        padded_image = Image.new('RGB', (target_size, target_size), pad_color)

        # Calculate padding offsets to center the image
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2

        # Paste resized image onto padded background
        padded_image.paste(resized_image, (paste_x, paste_y))

        return padded_image

    def __len__(self):
        return len(self.metadata)

    def load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image"""
        full_path = self.data_dir / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        image = Image.open(full_path).convert('RGB')

        # Apply random flip
        if random.random() < self.random_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def tokenize_caption(self, caption: str) -> Dict[str, torch.Tensor]:
        """Tokenize caption for both CLIP and T5 tokenizers"""
        # CLIP tokenization
        clip_inputs = self.clip_tokenizer(
            caption,
            padding="max_length",
            max_length=77,  # CLIP's max length
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        sample = self.metadata[idx]

        # Load image
        image_path = sample[self.image_column]
        image = self.load_image(image_path)

        # Transform image
        image_tensor = self.image_transform(image)

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
        }


class FluxDreamboothDataset(Dataset):
    """
    Dataset for FLUX.1-dev Dreambooth training
    Supports both instance and class images
    """

    def __init__(
        self,
        instance_data_dir: str,
        class_data_dir: Optional[str] = None,
        instance_prompt: str = "a photo of sks person",
        class_prompt: str = "a photo of person",
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: float = 0.0,
        tokenizer_name: str = "openai/clip-vit-large-patch14",
        t5_tokenizer_name: str = "google/t5-v1_1-xxl",
    ):
        self.instance_data_dir = Path(instance_data_dir)
        self.class_data_dir = Path(class_data_dir) if class_data_dir else None
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip

        # Setup tokenizers
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_tokenizer_name)

        # Load instance images
        self.instance_images = self._load_images(self.instance_data_dir)

        # Load class images if provided
        self.class_images = []
        if self.class_data_dir and self.class_data_dir.exists():
            self.class_images = self._load_images(self.class_data_dir)

        # Setup transforms
        self.setup_transforms()

        print(f"Loaded {len(self.instance_images)} instance images")
        if self.class_images:
            print(f"Loaded {len(self.class_images)} class images")

    def _load_images(self, data_dir: Path) -> List[Path]:
        """Load all images from directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = []

        for ext in image_extensions:
            images.extend(data_dir.glob(f"*{ext}"))
            images.extend(data_dir.glob(f"*{ext.upper()}"))

        return sorted(images)

    def setup_transforms(self):
        """Setup image transforms"""
        transform_list = []

        if self.center_crop:
            transform_list.extend([
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution),
            ])
        else:
            transform_list.append(
                transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BILINEAR)
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.image_transform = transforms.Compose(transform_list)

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
            max_length=512,
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
        return max(len(self.instance_images), len(self.class_images))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        # Determine if this is an instance or class sample
        if idx < len(self.instance_images):
            # Instance sample
            image_path = self.instance_images[idx]
            prompt = self.instance_prompt
            is_instance = True
        else:
            # Class sample (if available)
            if self.class_images:
                class_idx = idx % len(self.class_images)
                image_path = self.class_images[class_idx]
                prompt = self.class_prompt
                is_instance = False
            else:
                # Fallback to instance if no class images
                instance_idx = idx % len(self.instance_images)
                image_path = self.instance_images[instance_idx]
                prompt = self.instance_prompt
                is_instance = True

        # Load and transform image
        image = Image.open(image_path).convert('RGB')

        if random.random() < self.random_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_tensor = self.image_transform(image)

        # Tokenize prompt
        tokenized = self.tokenize_caption(prompt)

        return {
            "pixel_values": image_tensor,
            "input_ids_clip": tokenized["clip_input_ids"],
            "attention_mask_clip": tokenized["clip_attention_mask"],
            "input_ids_t5": tokenized["t5_input_ids"],
            "attention_mask_t5": tokenized["t5_attention_mask"],
            "caption": prompt,
            "image_path": str(image_path),
            "is_instance": is_instance,
        }


def create_flux_metadata_template(data_dir: str, images_subdir: str = "images"):
    """
    Create a metadata.json template for FLUX text-to-image dataset

    Args:
        data_dir: Root directory containing images
        images_subdir: Subdirectory containing images
    """
    data_path = Path(data_dir)
    images_path = data_path / images_subdir

    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))
        image_files.extend(images_path.glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)

    # Create metadata
    metadata = []
    for i, img_file in enumerate(image_files):
        # Try to extract caption from filename
        caption = img_file.stem.replace('_', ' ').replace('-', ' ')

        metadata.append({
            "image": f"{images_subdir}/{img_file.name}",
            "caption": f"A photo of {caption}"  # Add your custom caption here
        })

    # Save metadata
    metadata_path = data_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata template at {metadata_path}")
    print(f"Found {len(metadata)} images")
    print("Please edit the metadata.json file to add proper captions for your images")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--create_metadata", action="store_true", help="Create metadata template")
    parser.add_argument("--dataset_type", type=str, choices=["text2image", "dreambooth"], default="text2image")
    args = parser.parse_args()

    if args.create_metadata:
        create_flux_metadata_template(args.data_dir)
    else:
        # Test dataset loading
        if args.dataset_type == "text2image":
            dataset = FluxTextToImageDataset(args.data_dir)
        else:
            dataset = FluxDreamboothDataset(args.data_dir)

        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['pixel_values'].shape}")
            print(f"Caption: {sample['caption']}")
