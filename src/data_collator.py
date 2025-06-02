"""
Custom data collators for variable-sized image datasets
Handles batching of images with different aspect ratios efficiently
"""

import torch
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class VariableSizeCollator:
    """
    Data collator for variable-sized images that groups by bucket
    """
    
    def __init__(self, pad_to_max: bool = False):
        self.pad_to_max = pad_to_max
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch of variable-sized samples
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Collated batch with proper padding/grouping
        """
        if not batch:
            return {}
            
        # Group by bucket size for efficient batching
        buckets = {}
        for i, sample in enumerate(batch):
            bucket_id = sample.get("bucket_id", "default")
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append((i, sample))
            
        # If all samples are from the same bucket, batch normally
        if len(buckets) == 1:
            return self._collate_same_size_batch(batch)
        
        # If mixed buckets, either pad to max or return largest bucket
        if self.pad_to_max:
            return self._collate_padded_batch(batch)
        else:
            # Return the largest bucket
            largest_bucket = max(buckets.values(), key=len)
            largest_batch = [sample for _, sample in largest_bucket]
            return self._collate_same_size_batch(largest_batch)
            
    def _collate_same_size_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch where all samples have the same size"""
        collated = {}
        
        for key in batch[0].keys():
            if key in ["image_path", "video_path", "caption", "bucket_id"]:
                # String fields - just collect as list
                collated[key] = [sample[key] for sample in batch]
            elif key in ["target_width", "target_height", "target_size"]:
                # Size fields - collect as list or tensor
                values = [sample[key] for sample in batch]
                collated[key] = torch.tensor(values) if isinstance(values[0], (int, float)) else values
            else:
                # Tensor fields - stack
                tensors = [sample[key] for sample in batch]
                if isinstance(tensors[0], torch.Tensor):
                    try:
                        collated[key] = torch.stack(tensors, dim=0)
                    except RuntimeError as e:
                        # If stacking fails due to size mismatch, pad tensors
                        collated[key] = self._pad_and_stack_tensors(tensors)
                else:
                    collated[key] = tensors
                    
        return collated
        
    def _collate_padded_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch with padding to handle different sizes"""
        collated = {}
        
        # Find maximum dimensions for each tensor field
        max_dims = {}
        tensor_keys = []
        
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                tensor_keys.append(key)
                shapes = [sample[key].shape for sample in batch]
                max_dims[key] = tuple(max(dim) for dim in zip(*shapes))
                
        # Collate non-tensor fields
        for key in batch[0].keys():
            if key not in tensor_keys:
                if key in ["image_path", "video_path", "caption", "bucket_id"]:
                    collated[key] = [sample[key] for sample in batch]
                elif key in ["target_width", "target_height", "target_size"]:
                    values = [sample[key] for sample in batch]
                    collated[key] = torch.tensor(values) if isinstance(values[0], (int, float)) else values
                else:
                    collated[key] = [sample[key] for sample in batch]
                    
        # Pad and collate tensor fields
        for key in tensor_keys:
            tensors = [sample[key] for sample in batch]
            padded_tensors = []
            
            for tensor in tensors:
                padded = self._pad_tensor_to_shape(tensor, max_dims[key])
                padded_tensors.append(padded)
                
            collated[key] = torch.stack(padded_tensors, dim=0)
            
        return collated
        
    def _pad_tensor_to_shape(self, tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Pad tensor to target shape"""
        current_shape = tensor.shape
        
        if current_shape == target_shape:
            return tensor
            
        # Calculate padding for each dimension
        padding = []
        for i in range(len(current_shape) - 1, -1, -1):  # Reverse order for F.pad
            diff = target_shape[i] - current_shape[i]
            padding.extend([0, diff])
            
        # Apply padding
        padded = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
        return padded
        
    def _pad_and_stack_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Pad tensors to same shape and stack"""
        if not tensors:
            return torch.empty(0)
            
        # Find maximum shape
        shapes = [t.shape for t in tensors]
        max_shape = tuple(max(dim) for dim in zip(*shapes))
        
        # Pad all tensors to max shape
        padded_tensors = []
        for tensor in tensors:
            padded = self._pad_tensor_to_shape(tensor, max_shape)
            padded_tensors.append(padded)
            
        return torch.stack(padded_tensors, dim=0)


class BucketBatchSampler:
    """
    Batch sampler that groups samples by bucket for efficient training
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        bucket_key: str = "bucket_id"
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.bucket_key = bucket_key
        
        # Group indices by bucket
        self.bucket_indices = {}
        for idx in range(len(dataset)):
            sample = dataset[idx]
            bucket_id = sample.get(bucket_key, "default")
            if bucket_id not in self.bucket_indices:
                self.bucket_indices[bucket_id] = []
            self.bucket_indices[bucket_id].append(idx)
            
        print(f"Created buckets: {[(k, len(v)) for k, v in self.bucket_indices.items()]}")
        
    def __iter__(self):
        """Generate batches grouped by bucket"""
        all_batches = []
        
        for bucket_id, indices in self.bucket_indices.items():
            if self.shuffle:
                np.random.shuffle(indices)
                
            # Create batches for this bucket
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    all_batches.append(batch_indices)
                    
        # Shuffle batches if requested
        if self.shuffle:
            np.random.shuffle(all_batches)
            
        for batch in all_batches:
            yield batch
            
    def __len__(self):
        """Return number of batches"""
        total_batches = 0
        for indices in self.bucket_indices.values():
            bucket_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                bucket_batches += 1
            total_batches += bucket_batches
        return total_batches


class AspectRatioGroupedSampler:
    """
    Sampler that groups images by aspect ratio for efficient batching
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        aspect_ratio_tolerance: float = 0.1,
        drop_last: bool = False,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Calculate aspect ratios and group indices
        self.aspect_ratio_groups = self._group_by_aspect_ratio()
        
    def _group_by_aspect_ratio(self):
        """Group dataset indices by aspect ratio"""
        aspect_ratios = []
        
        # Calculate aspect ratios for all samples
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            
            if "target_width" in sample and "target_height" in sample:
                width = sample["target_width"]
                height = sample["target_height"]
                aspect_ratio = width / height
            else:
                # Default aspect ratio
                aspect_ratio = 16 / 9
                
            aspect_ratios.append((idx, aspect_ratio))
            
        # Sort by aspect ratio
        aspect_ratios.sort(key=lambda x: x[1])
        
        # Group by similar aspect ratios
        groups = []
        current_group = []
        current_ratio = None
        
        for idx, ratio in aspect_ratios:
            if current_ratio is None or abs(ratio - current_ratio) <= self.aspect_ratio_tolerance:
                current_group.append(idx)
                current_ratio = ratio
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [idx]
                current_ratio = ratio
                
        if current_group:
            groups.append(current_group)
            
        print(f"Grouped {len(self.dataset)} samples into {len(groups)} aspect ratio groups")
        return groups
        
    def __iter__(self):
        """Generate batches grouped by aspect ratio"""
        all_batches = []
        
        for group in self.aspect_ratio_groups:
            if self.shuffle:
                np.random.shuffle(group)
                
            # Create batches for this group
            for i in range(0, len(group), self.batch_size):
                batch_indices = group[i:i + self.batch_size]
                
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    all_batches.append(batch_indices)
                    
        # Shuffle batches if requested
        if self.shuffle:
            np.random.shuffle(all_batches)
            
        for batch in all_batches:
            yield batch
            
    def __len__(self):
        """Return number of batches"""
        total_batches = 0
        for group in self.aspect_ratio_groups:
            group_batches = len(group) // self.batch_size
            if not self.drop_last and len(group) % self.batch_size > 0:
                group_batches += 1
            total_batches += group_batches
        return total_batches


def create_variable_size_dataloader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    use_bucket_sampler: bool = True,
    use_aspect_ratio_grouping: bool = False,
    **kwargs
):
    """
    Create a DataLoader optimized for variable-sized images
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        use_bucket_sampler: Whether to use bucket-based sampling
        use_aspect_ratio_grouping: Whether to group by aspect ratio
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    # Choose sampler
    sampler = None
    batch_sampler = None
    
    if use_bucket_sampler and hasattr(dataset, 'sample_buckets'):
        batch_sampler = BucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle
        )
        shuffle = False  # Handled by batch sampler
    elif use_aspect_ratio_grouping:
        batch_sampler = AspectRatioGroupedSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle
        )
        shuffle = False  # Handled by batch sampler
        
    # Create collator
    collate_fn = VariableSizeCollator(pad_to_max=True)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size if batch_sampler is None else 1,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **kwargs
    )
    
    return dataloader
