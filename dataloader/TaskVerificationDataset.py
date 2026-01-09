"""
Dataset class for Task Verification

Handles step-level embeddings with variable-length sequences
and binary video-level labels.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TaskVerificationDataset(Dataset):
    """
    Dataset for task verification with step embeddings.
    
    Each sample consists of:
        - A sequence of step embeddings (variable length, padded)
        - A binary label (0: no errors, 1: has errors)
        - A mask indicating which steps are real vs. padding
        - The recording ID for tracking
    
    Args:
        embeddings: np.array of shape (N, max_steps, embed_dim)
            Step embeddings for all videos, padded to max_steps
        labels: np.array of shape (N,)
            Binary labels (0 or 1)
        masks: np.array of shape (N, max_steps)
            Boolean masks (True for real steps, False for padding)
        recording_ids: list of str
            Recording IDs for each video
    """
    
    def __init__(self, embeddings, labels, masks, recording_ids):
        """Initialize the dataset."""
        assert len(embeddings) == len(labels) == len(masks) == len(recording_ids), \
            "All inputs must have the same length"
        
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)  # (N,) -> (N, 1)
        self.masks = torch.BoolTensor(masks)
        self.recording_ids = recording_ids
        
        # Store statistics
        self.num_samples = len(embeddings)
        self.max_steps = embeddings.shape[1]
        self.embed_dim = embeddings.shape[2]
        self.num_positive = int(labels.sum())
        self.num_negative = self.num_samples - self.num_positive
        
    def __len__(self):
        """Return the number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary containing:
                - embeddings: (max_steps, embed_dim)
                - label: (1,) binary label
                - mask: (max_steps,) boolean mask
                - recording_id: str
        """
        return {
            'embeddings': self.embeddings[idx],
            'label': self.labels[idx],
            'mask': self.masks[idx],
            'recording_id': self.recording_ids[idx]
        }
    
    def get_class_weights(self):
        """
        Compute class weights for handling imbalanced data.
        
        Returns:
            weights: torch.Tensor of shape (2,)
                Weights for [negative_class, positive_class]
        """
        if self.num_positive == 0 or self.num_negative == 0:
            return torch.tensor([1.0, 1.0])
        
        # Inverse frequency weighting
        weight_negative = self.num_samples / (2.0 * self.num_negative)
        weight_positive = self.num_samples / (2.0 * self.num_positive)
        
        return torch.tensor([weight_negative, weight_positive])
    
    def get_statistics(self):
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        # Calculate actual number of steps per video (from mask)
        actual_steps = self.masks.sum(dim=1).numpy()
        
        stats = {
            'num_samples': self.num_samples,
            'num_positive': self.num_positive,
            'num_negative': self.num_negative,
            'positive_ratio': self.num_positive / self.num_samples,
            'max_steps': self.max_steps,
            'embed_dim': self.embed_dim,
            'avg_steps_per_video': actual_steps.mean(),
            'min_steps_per_video': actual_steps.min(),
            'max_steps_per_video': actual_steps.max(),
        }
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics in a readable format."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Task Verification Dataset Statistics")
        print("="*60)
        print(f"Total samples:        {stats['num_samples']}")
        print(f"Positive (errors):    {stats['num_positive']} ({stats['positive_ratio']*100:.1f}%)")
        print(f"Negative (no errors): {stats['num_negative']} ({(1-stats['positive_ratio'])*100:.1f}%)")
        print(f"\nEmbedding dimension:  {stats['embed_dim']}")
        print(f"Max steps (padded):   {stats['max_steps']}")
        print(f"Avg steps per video:  {stats['avg_steps_per_video']:.1f}")
        print(f"Min steps per video:  {stats['min_steps_per_video']}")
        print(f"Max steps per video:  {stats['max_steps_per_video']}")
        print("="*60 + "\n")


def collate_task_verification(batch):
    """
    Custom collate function for TaskVerificationDataset.
    
    This is useful if you need custom batching logic.
    For now, it just uses default batching.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched dictionary
    """
    embeddings = torch.stack([item['embeddings'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    recording_ids = [item['recording_id'] for item in batch]
    
    return {
        'embeddings': embeddings,
        'label': labels,
        'mask': masks,
        'recording_id': recording_ids
    }
