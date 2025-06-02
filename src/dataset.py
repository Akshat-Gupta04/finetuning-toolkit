"""
Dataset utilities for Wan2.1 Image-to-Video finetuning
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor
import decord
from decord import VideoReader, cpu
import imageio

decord.bridge.set_bridge('torch')


class ImageVideoDataset(Dataset):
    """
    Dataset for Image-to-Video training with Wan2.1

    Expected data structure:
    data_dir/
    ├── metadata.json  # Contains image_path, video_path, caption for each sample
    ├── images/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── videos/
        ├── video1.mp4
        └── video2.mp4
    """

    def __init__(
        self,
        data_dir: str,
        image_column: str = "image",
        video_column: str = "video",
        caption_column: str = "caption",
        width: int = 1280,
        height: int = 720,
        num_frames: int = 81,
        frame_rate: int = 16,
        sample_stride: int = 1,
        random_flip: float = 0.0,
        center_crop: bool = True,
        normalize: bool = True,
        max_sequence_length: int = 256,
    ):
        self.data_dir = Path(data_dir)
        self.image_column = image_column
        self.video_column = video_column
        self.caption_column = caption_column

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.sample_stride = sample_stride
        self.random_flip = random_flip
        self.center_crop = center_crop
        self.normalize = normalize
        self.max_sequence_length = max_sequence_length

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Setup transforms
        self.image_processor = CLIPImageProcessor()
        self.setup_transforms()

    def setup_transforms(self):
        """Setup image and video transforms with aspect ratio preservation"""
        # We'll handle resizing manually to preserve aspect ratios
        # Only basic transforms here
        basic_transforms = [transforms.ToTensor()]

        if self.normalize:
            basic_transforms.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )

        self.image_transform = transforms.Compose(basic_transforms)
        self.video_transform = transforms.Compose(basic_transforms)

    def resize_with_aspect_ratio(self, image, target_width, target_height, pad_color=(0, 0, 0)):
        """
        Resize image while maintaining aspect ratio and pad to target size

        Args:
            image: PIL Image
            target_width: Target width
            target_height: Target height
            pad_color: Color for padding (RGB tuple)

        Returns:
            PIL Image resized and padded to target size
        """
        original_width, original_height = image.size

        # Calculate scaling factor to fit within target dimensions
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)

        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image maintaining aspect ratio
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target dimensions and pad color
        padded_image = Image.new('RGB', (target_width, target_height), pad_color)

        # Calculate padding offsets to center the image
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        # Paste resized image onto padded background
        padded_image.paste(resized_image, (paste_x, paste_y))

        return padded_image

    def __len__(self):
        return len(self.metadata)

    def load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image with aspect ratio preservation"""
        full_path = self.data_dir / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        image = Image.open(full_path).convert('RGB')

        # Apply random flip
        if random.random() < self.random_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize with aspect ratio preservation
        if self.center_crop:
            # For center crop mode, resize to fit and then crop
            image = self.resize_with_aspect_ratio(image, self.width, self.height)
        else:
            # For non-crop mode, resize to fit with padding
            image = self.resize_with_aspect_ratio(image, self.width, self.height)

        return image

    def load_video(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video"""
        full_path = self.data_dir / video_path
        if not full_path.exists():
            raise FileNotFoundError(f"Video not found: {full_path}")

        try:
            # Use decord for efficient video loading
            vr = VideoReader(str(full_path), ctx=cpu(0))
            total_frames = len(vr)

            # Sample frames
            if total_frames < self.num_frames:
                # Repeat frames if video is too short
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                # Sample frames with stride
                start_idx = random.randint(0, max(0, total_frames - self.num_frames * self.sample_stride))
                frame_indices = np.arange(start_idx, start_idx + self.num_frames * self.sample_stride, self.sample_stride)
                frame_indices = frame_indices[:self.num_frames]

            # Load frames
            frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)

            # Convert to PIL Images and apply transforms with aspect ratio preservation
            video_frames = []
            for frame in frames:
                frame_pil = Image.fromarray(frame)

                # Apply random flip (same as image)
                if random.random() < self.random_flip:
                    frame_pil = frame_pil.transpose(Image.FLIP_LEFT_RIGHT)

                # Resize with aspect ratio preservation
                frame_pil = self.resize_with_aspect_ratio(frame_pil, self.width, self.height)

                frame_tensor = self.video_transform(frame_pil)
                video_frames.append(frame_tensor)

            video_tensor = torch.stack(video_frames, dim=0)  # (T, C, H, W)

        except Exception as e:
            print(f"Error loading video {full_path}: {e}")
            # Fallback: create dummy video
            video_tensor = torch.zeros(self.num_frames, 3, self.height, self.width)

        return video_tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        sample = self.metadata[idx]

        # Load image
        image_path = sample[self.image_column]
        image = self.load_image(image_path)

        # Process image for CLIP encoder
        image_clip = self.image_processor(image, return_tensors="pt")["pixel_values"][0]

        # Transform image for VAE
        image_vae = self.image_transform(image)

        # Load video
        video_path = sample[self.video_column]
        video = self.load_video(video_path)

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
        }


def create_metadata_template(data_dir: str, images_subdir: str = "images", videos_subdir: str = "videos"):
    """
    Create a metadata.json template for the dataset

    Args:
        data_dir: Root directory containing images and videos
        images_subdir: Subdirectory containing images
        videos_subdir: Subdirectory containing videos
    """
    data_path = Path(data_dir)
    images_path = data_path / images_subdir
    videos_path = data_path / videos_subdir

    if not images_path.exists() or not videos_path.exists():
        raise FileNotFoundError(f"Images or videos directory not found in {data_dir}")

    # Get all image and video files
    image_files = sorted([f for f in images_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    video_files = sorted([f for f in videos_path.iterdir() if f.suffix.lower() in ['.mp4', '.avi', '.mov']])

    if len(image_files) != len(video_files):
        print(f"Warning: Number of images ({len(image_files)}) != number of videos ({len(video_files)})")

    # Create metadata
    metadata = []
    for i, (img_file, vid_file) in enumerate(zip(image_files, video_files)):
        metadata.append({
            "image": f"{images_subdir}/{img_file.name}",
            "video": f"{videos_subdir}/{vid_file.name}",
            "caption": f"Sample {i+1} - Add your caption here"
        })

    # Save metadata
    metadata_path = data_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata template at {metadata_path}")
    print(f"Found {len(metadata)} image-video pairs")
    print("Please edit the metadata.json file to add proper captions for your data")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--create_metadata", action="store_true", help="Create metadata template")
    args = parser.parse_args()

    if args.create_metadata:
        create_metadata_template(args.data_dir)
    else:
        # Test dataset loading
        dataset = ImageVideoDataset(args.data_dir)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Image CLIP shape: {sample['image_clip'].shape}")
            print(f"Image VAE shape: {sample['image_vae'].shape}")
            print(f"Video shape: {sample['video'].shape}")
            print(f"Caption: {sample['caption']}")
