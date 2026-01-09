"""
Task Verification Model for Recipe Execution Assessment

This module implements a transformer-based binary classification model
that predicts whether a recipe execution video contains errors.
"""

import torch
from torch import nn
from core.models.blocks import EncoderLayer, Encoder


class TaskVerifier(nn.Module):
    """
    Transformer-based model for task verification.
    Takes sequence of step embeddings and predicts if video has errors.
    
    Architecture:
        1. Transformer encoder to process step sequences
        2. Global pooling with masking
        3. Binary classification head
    
    Args:
        embedding_dim: Dimension of step embeddings (default: 1024)
        hidden_dim: Hidden dimension for classification head (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 1)
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(
        self,
        embedding_dim=1024,
        hidden_dim=512,
        num_heads=8,
        num_layers=1,
        dropout=0.3
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Transformer encoder to process step sequence
        encoder_layer = EncoderLayer(
            d_model=embedding_dim,
            dim_feedforward=2048,
            nhead=num_heads,
            batch_first=True
        )
        self.step_encoder = Encoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, step_embeddings, mask=None):
        """
        Forward pass through the task verification model.
        
        Args:
            step_embeddings: (batch_size, num_steps, embedding_dim)
                Sequence of step-level embeddings
            mask: (batch_size, num_steps) boolean mask, optional
                True for valid steps, False for padding
        
        Returns:
            predictions: (batch_size, 1) 
                Binary predictions (probabilities between 0 and 1)
        """
        # Handle NaN values in input
        step_embeddings = torch.nan_to_num(step_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Encode step sequence with transformer
        # This allows steps to attend to each other and capture temporal dependencies
        encoded = self.step_encoder(step_embeddings)  # (batch_size, num_steps, embedding_dim)
        
        # Global pooling: aggregate step information
        # Use mask to only average over real steps (not padding)
        if mask is not None:
            # Expand mask to match embedding dimensions
            mask = mask.unsqueeze(-1).float()  # (batch_size, num_steps, 1)
            
            # Masked average pooling
            # Multiply by mask to zero out padding positions
            masked_encoded = encoded * mask  # (batch_size, num_steps, embedding_dim)
            
            # Sum and divide by number of real steps
            pooled = masked_encoded.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch_size, embedding_dim)
        else:
            # Simple average pooling if no mask provided
            pooled = encoded.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Binary classification
        output = self.classifier(pooled)  # (batch_size, 1)
        
        return output
    
    def predict(self, step_embeddings, mask=None, threshold=0.5):
        """
        Make binary predictions (0 or 1) instead of probabilities.
        
        Args:
            step_embeddings: (batch_size, num_steps, embedding_dim)
            mask: (batch_size, num_steps) boolean mask, optional
            threshold: Decision threshold (default: 0.5)
        
        Returns:
            predictions: (batch_size, 1) binary predictions (0 or 1)
        """
        with torch.no_grad():
            probs = self.forward(step_embeddings, mask)
            preds = (probs > threshold).long()
        return preds
    
    def get_num_parameters(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleMLPVerifier(nn.Module):
    """
    Simple MLP baseline for task verification.
    Uses average pooling followed by MLP classification.
    
    This is a simpler alternative to the transformer-based TaskVerifier.
    """
    
    def __init__(self, embedding_dim=1024, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, step_embeddings, mask=None):
        """
        Forward pass with simple average pooling.
        
        Args:
            step_embeddings: (batch_size, num_steps, embedding_dim)
            mask: (batch_size, num_steps) boolean mask, optional
        
        Returns:
            predictions: (batch_size, 1)
        """
        # Handle NaN values
        step_embeddings = torch.nan_to_num(step_embeddings, nan=0.0)
        
        # Average pooling with mask
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            pooled = (step_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = step_embeddings.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        return output
    
    def get_num_parameters(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
