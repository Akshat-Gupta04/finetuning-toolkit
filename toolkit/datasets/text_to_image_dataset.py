"""
Text-to-Image dataset implementation for FLUX training
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from ..core import BaseDataset, DatasetConfig


class TextToImageDataset(BaseDataset):
    """Dataset for text-to-image training (FLUX)"""
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.image_transforms = self._setup_transforms()
        
    def prepare_data(self) -> None:
        """Prepare text-to-image dataset"""
        data_dir = Path(self.config.data_dir)
        
        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        # Look for metadata file
        metadata_file = data_dir / "metadata.jsonl"
        if not metadata_file.exists():
            metadata_file = data_dir / "metadata.json"
        
        if metadata_file.exists():
            self._load_from_metadata(metadata_file)
        else:
            self._load_from_directory(data_dir)
        
        logging.info(f"Loaded {len(self.data)} text-image pairs")
    
    def _load_from_metadata(self, metadata_file: Path) -> None:
        """Load dataset from metadata file"""
        data_dir = metadata_file.parent
        
        if metadata_file.suffix == ".jsonl":
            # JSONL format
            with open(metadata_file, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    self._add_item(item, data_dir)
        else:
            # JSON format
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if isinstance(metadata, list):
                    for item in metadata:
                        self._add_item(item, data_dir)
                else:
                    # Single item
                    self._add_item(metadata, data_dir)
    
    def _load_from_directory(self, data_dir: Path) -> None:
        """Load dataset by scanning directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for image_path in data_dir.rglob("*"):
            if image_path.suffix.lower() in image_extensions:
                # Look for corresponding caption file
                caption_path = image_path.with_suffix('.txt')
                if caption_path.exists():
                    with open(caption_path, 'r') as f:
                        caption = f.read().strip()
                    
                    self.data.append({
                        'image_path': str(image_path),
                        'caption': caption
                    })
    
    def _add_item(self, item: Dict[str, Any], data_dir: Path) -> None:
        """Add item to dataset"""
        image_column = self.config.image_column
        caption_column = self.config.caption_column
        
        if image_column not in item or caption_column not in item:
            logging.warning(f"Skipping item missing required columns: {item}")
            return
        
        image_path = data_dir / item[image_column]
        if not image_path.exists():
            logging.warning(f"Image file not found: {image_path}")
            return
        
        self.data.append({
            'image_path': str(image_path),
            'caption': item[caption_column]
        })
    
    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transforms"""
        transform_list = []
        
        # Resize
        if isinstance(self.config.resolution, int):
            size = (self.config.resolution, self.config.resolution)
        else:
            size = self.config.resolution
        
        if self.config.center_crop:
            transform_list.append(transforms.CenterCrop(size))
        else:
            transform_list.append(transforms.Resize(size))
        
        # Random flip
        if self.config.random_flip > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=self.config.random_flip))
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        return transforms.Compose(transform_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index"""
        item = self.data[idx]
        
        # Load and transform image
        try:
            image = Image.open(item['image_path']).convert('RGB')
            image = self.image_transforms(image)
        except Exception as e:
            logging.error(f"Error loading image {item['image_path']}: {e}")
            # Return a black image as fallback
            if isinstance(self.config.resolution, int):
                size = (3, self.config.resolution, self.config.resolution)
            else:
                size = (3, self.config.resolution[1], self.config.resolution[0])
            image = torch.zeros(size)
        
        return {
            'pixel_values': image,
            'caption': item['caption'],
            'image_path': item['image_path']
        }
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function"""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        captions = [item['caption'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        
        return {
            'pixel_values': pixel_values,
            'caption': captions,
            'image_path': image_paths
        }


class VariableSizeTextToImageDataset(TextToImageDataset):
    """Variable-size text-to-image dataset for aspect ratio preservation"""
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.bucket_sizes = config.bucket_sizes or self._generate_bucket_sizes()
        self.image_buckets = self._create_buckets()
        
    def _generate_bucket_sizes(self) -> List[tuple]:
        """Generate bucket sizes for different aspect ratios"""
        base_resolution = self.config.resolution if isinstance(self.config.resolution, int) else 512
        
        # Common aspect ratios
        aspect_ratios = [
            (1, 1),    # Square
            (4, 3),    # 4:3
            (3, 4),    # 3:4
            (16, 9),   # 16:9
            (9, 16),   # 9:16
            (3, 2),    # 3:2
            (2, 3),    # 2:3
        ]
        
        bucket_sizes = []
        for w_ratio, h_ratio in aspect_ratios:
            # Calculate dimensions that maintain aspect ratio and target pixel count
            target_pixels = base_resolution * base_resolution
            scale = (target_pixels / (w_ratio * h_ratio)) ** 0.5
            width = int(w_ratio * scale)
            height = int(h_ratio * scale)
            
            # Round to multiples of 8 for better performance
            width = (width // 8) * 8
            height = (height // 8) * 8
            
            bucket_sizes.append((width, height))
        
        return bucket_sizes
    
    def _create_buckets(self) -> Dict[tuple, List[int]]:
        """Create buckets based on image aspect ratios"""
        buckets = {size: [] for size in self.bucket_sizes}
        
        for idx, item in enumerate(self.data):
            try:
                with Image.open(item['image_path']) as img:
                    aspect_ratio = img.width / img.height
                    
                    # Find best matching bucket
                    best_bucket = min(
                        self.bucket_sizes,
                        key=lambda size: abs(size[0] / size[1] - aspect_ratio)
                    )
                    buckets[best_bucket].append(idx)
            except Exception as e:
                logging.warning(f"Error processing image {item['image_path']}: {e}")
                # Default to square bucket
                buckets[self.bucket_sizes[0]].append(idx)
        
        # Remove empty buckets
        buckets = {size: indices for size, indices in buckets.items() if indices}
        
        logging.info(f"Created {len(buckets)} buckets with sizes: {list(buckets.keys())}")
        return buckets
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index with appropriate bucket size"""
        item = self.data[idx]
        
        # Find which bucket this item belongs to
        bucket_size = None
        for size, indices in self.image_buckets.items():
            if idx in indices:
                bucket_size = size
                break
        
        if bucket_size is None:
            bucket_size = self.bucket_sizes[0]  # Default to first bucket
        
        # Load and transform image to bucket size
        try:
            image = Image.open(item['image_path']).convert('RGB')
            
            # Resize to bucket size
            transform = transforms.Compose([
                transforms.Resize(bucket_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image = transform(image)
        except Exception as e:
            logging.error(f"Error loading image {item['image_path']}: {e}")
            # Return a black image as fallback
            image = torch.zeros((3, bucket_size[1], bucket_size[0]))
        
        return {
            'pixel_values': image,
            'caption': item['caption'],
            'image_path': item['image_path'],
            'bucket_size': bucket_size
        }
    
    def get_bucket_batch(self, bucket_size: tuple, batch_size: int) -> List[int]:
        """Get a batch of indices from a specific bucket"""
        if bucket_size not in self.image_buckets:
            return []
        
        indices = self.image_buckets[bucket_size]
        if len(indices) < batch_size:
            # Repeat indices if not enough samples
            indices = indices * (batch_size // len(indices) + 1)
        
        return indices[:batch_size]
