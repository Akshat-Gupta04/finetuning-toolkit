"""
Automatic dataset preparation for diffusion model training
Supports image and video captioning with AI models
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from PIL import Image, ImageOps
import numpy as np
from tqdm.auto import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Video processing will be limited.")

try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logging.warning("Decord not available. Video processing will be limited.")


class AutoDatasetProcessor:
    """Unified automatic dataset processor for diffusion training"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_type: str = "flux",  # "flux" or "wan2.1"
        captioning_model: str = "blip2",
        min_resolution: int = 512,
        max_resolution: int = 2048,
        quality_threshold: float = 0.7,
        batch_size: int = 8,  # Optimized for A40
        num_workers: int = 8,
        max_images: Optional[int] = None,
        device: str = "cuda"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_type = model_type.lower()
        self.captioning_model = captioning_model
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.quality_threshold = quality_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_images = max_images
        self.device = device

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize captioner
        self.captioner = None

        # Supported formats
        self.image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

        # Determine if we need video support
        self.include_videos = model_type in ["wan2_1_i2v", "wan2_1_t2v"]

        # Statistics
        self.stats = {
            "total_found": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "captions_generated": 0
        }

    def setup_captioner(self):
        """Initialize captioning model optimized for A40"""
        if self.captioner is not None:
            return

        self.logger.info(f"Loading {self.captioning_model} on {self.device}")

        try:
            if self.captioning_model == "blip2":
                self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:  # blip
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )

            self.model.to(self.device)
            self.model.eval()
            self.captioner = True

        except Exception as e:
            self.logger.error(f"Failed to load captioning model: {e}")
            raise

    def find_media_files(self) -> List[Path]:
        """Find all valid images and videos"""
        media_files = []

        for file_path in self.input_dir.rglob("*"):
            suffix = file_path.suffix.lower()
            if suffix in self.image_formats:
                media_files.append(file_path)
            elif self.include_videos and suffix in self.video_formats:
                media_files.append(file_path)

        media_files.sort()

        if self.max_images:
            media_files = media_files[:self.max_images]

        self.stats["total_found"] = len(media_files)
        self.logger.info(f"Found {len(media_files)} media files")

        return media_files

    def extract_video_frames(self, video_path: Path, num_frames: int = 8) -> List[Image.Image]:
        """Extract frames from video for captioning"""
        frames = []

        if DECORD_AVAILABLE:
            try:
                vr = decord.VideoReader(str(video_path))
                total_frames = len(vr)

                # Sample frames evenly
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

                for idx in frame_indices:
                    frame = vr[idx].asnumpy()
                    # Convert BGR to RGB if needed
                    if frame.shape[-1] == 3:  # RGB/BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else frame
                    frames.append(Image.fromarray(frame))

            except Exception as e:
                self.logger.error(f"Failed to extract frames with decord: {e}")

        elif CV2_AVAILABLE:
            try:
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Sample frames evenly
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame))

                cap.release()

            except Exception as e:
                self.logger.error(f"Failed to extract frames with cv2: {e}")

        return frames

    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Analyze image quality and properties"""
        try:
            with Image.open(image_path) as img:
                # Handle EXIF orientation
                img = ImageOps.exif_transpose(img)
                width, height = img.size

                # Quality metrics
                min_dim = min(width, height)
                max_dim = max(width, height)
                aspect_ratio = width / height

                # Quality scoring
                resolution_score = min(min_dim / self.min_resolution, 1.0)
                aspect_score = 1.0 if 0.5 <= aspect_ratio <= 2.0 else 0.7
                quality_score = (resolution_score + aspect_score) / 2

                # Validation
                valid = (
                    quality_score >= self.quality_threshold and
                    min_dim >= self.min_resolution and
                    max_dim <= self.max_resolution * 2  # Allow some flexibility
                )

                return {
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                    "quality_score": quality_score,
                    "valid": valid,
                    "file_size": image_path.stat().st_size
                }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def caption_image(self, image: Image.Image) -> str:
        """Generate caption for single image"""
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                if self.captioning_model == "blip2":
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=5,
                        temperature=1.0,
                        do_sample=False
                    )
                else:  # blip
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=5,
                        temperature=1.0
                    )

            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)

            # Clean up caption
            if self.captioning_model == "blip2" and caption.startswith("a photo of"):
                caption = caption[10:].strip()

            return caption.strip() or "an image"

        except Exception as e:
            self.logger.error(f"Captioning error: {e}")
            return "an image"

    def caption_batch(self, images: List[Image.Image]) -> List[str]:
        """Generate captions for batch of images"""
        captions = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_captions = [self.caption_image(img) for img in batch]
            captions.extend(batch_captions)

        return captions

    def process_image(self, image_path: Path, output_images_dir: Path, index: int) -> Optional[Dict[str, Any]]:
        """Process single image: analyze, copy, prepare metadata"""
        try:
            # Analyze image
            analysis = self.analyze_image(image_path)

            if not analysis.get("valid", False):
                self.stats["skipped"] += 1
                return None

            # Generate output filename
            output_filename = f"image_{index:06d}{image_path.suffix.lower()}"
            output_path = output_images_dir / output_filename

            # Copy/resize image
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)

                # Resize if too large
                if max(img.size) > self.max_resolution:
                    img.thumbnail((self.max_resolution, self.max_resolution), Image.Resampling.LANCZOS)

                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Save
                img.save(output_path, quality=95, optimize=True)

            # Prepare metadata
            metadata = {
                "image": f"images/{output_filename}",
                "width": analysis["width"],
                "height": analysis["height"],
                "aspect_ratio": analysis["aspect_ratio"],
                "quality_score": analysis["quality_score"],
                "original_path": str(image_path)
            }

            self.stats["processed"] += 1
            return metadata

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Error processing {image_path}: {e}")
            return None

    def generate_captions(self, metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate captions for all processed images"""
        if not metadata_list:
            return metadata_list

        self.logger.info(f"Generating captions for {len(metadata_list)} images")
        self.setup_captioner()

        # Load images
        images = []
        valid_metadata = []

        for metadata in tqdm(metadata_list, desc="Loading images"):
            try:
                image_path = self.output_dir / metadata["image"]
                with Image.open(image_path) as img:
                    images.append(img.convert("RGB"))
                    valid_metadata.append(metadata)
            except Exception as e:
                self.logger.error(f"Error loading {metadata['image']}: {e}")

        # Generate captions
        captions = self.caption_batch(images)

        # Add captions to metadata
        for metadata, caption in zip(valid_metadata, captions):
            metadata["caption"] = caption

        self.stats["captions_generated"] = len(captions)
        return valid_metadata

    def create_dataset_structure(self):
        """Create output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        return images_dir

    def save_metadata(self, metadata_list: List[Dict[str, Any]]):
        """Save metadata in appropriate format"""
        metadata_path = self.output_dir / "metadata.json"

        # Format metadata based on model type
        if self.model_type == "flux":
            # FLUX format: simple image + caption
            formatted_metadata = [
                {
                    "image": item["image"],
                    "caption": item["caption"]
                }
                for item in metadata_list
            ]
        else:
            # Wan2.1 format: keep all fields for video training
            formatted_metadata = metadata_list

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved metadata to {metadata_path}")

    def process_dataset(self) -> Dict[str, Any]:
        """Main processing pipeline"""
        start_time = time.time()

        self.logger.info(f"Processing dataset for {self.model_type.upper()} training")

        # Create structure
        images_dir = self.create_dataset_structure()

        # Find images
        image_files = self.find_images()
        if not image_files:
            raise ValueError("No valid images found")

        # Process images
        self.logger.info("Processing images...")
        metadata_list = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.process_image, img_path, images_dir, i): i
                for i, img_path in enumerate(image_files)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                if result:
                    metadata_list.append(result)

        if not metadata_list:
            raise ValueError("No images were successfully processed")

        # Generate captions
        metadata_list = self.generate_captions(metadata_list)

        # Save metadata
        self.save_metadata(metadata_list)

        # Save statistics
        processing_time = time.time() - start_time
        self.stats.update({
            "processing_time": processing_time,
            "model_type": self.model_type,
            "captioning_model": self.captioning_model
        })

        stats_path = self.output_dir / "processing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        # Summary
        self.logger.info(f"Dataset processing completed in {processing_time:.1f}s")
        self.logger.info(f"Processed: {self.stats['processed']} images")
        self.logger.info(f"Generated: {self.stats['captions_generated']} captions")
        self.logger.info(f"Output: {self.output_dir}")

        return self.stats


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    model_type: str = "flux",
    captioning_model: str = "blip2",
    min_resolution: int = 512,
    max_resolution: int = 2048,
    quality_threshold: float = 0.7,
    batch_size: int = 8,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    """
    Unified function to prepare datasets for diffusion training

    Args:
        input_dir: Directory containing raw images
        output_dir: Output directory for processed dataset
        model_type: "flux" or "wan2.1"
        captioning_model: "blip2" or "blip"
        min_resolution: Minimum image resolution
        max_resolution: Maximum image resolution
        quality_threshold: Quality threshold (0-1)
        batch_size: Batch size for processing
        max_images: Maximum number of images to process

    Returns:
        Processing statistics
    """
    processor = AutoDatasetProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        model_type=model_type,
        captioning_model=captioning_model,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        quality_threshold=quality_threshold,
        batch_size=batch_size,
        max_images=max_images
    )

    return processor.process_dataset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automatic dataset preparation")
    parser.add_argument("--input_dir", required=True, help="Input directory with images")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_type", default="flux", choices=["flux", "wan2.1"], help="Model type")
    parser.add_argument("--captioning_model", default="blip2", choices=["blip2", "blip"], help="Captioning model")
    parser.add_argument("--min_resolution", type=int, default=512, help="Minimum resolution")
    parser.add_argument("--max_resolution", type=int, default=2048, help="Maximum resolution")
    parser.add_argument("--quality_threshold", type=float, default=0.7, help="Quality threshold")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_images", type=int, help="Maximum images to process")

    args = parser.parse_args()

    stats = prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        captioning_model=args.captioning_model,
        min_resolution=args.min_resolution,
        max_resolution=args.max_resolution,
        quality_threshold=args.quality_threshold,
        batch_size=args.batch_size,
        max_images=args.max_images
    )

    print(f"✅ Dataset preparation completed: {stats['processed']} images processed")
