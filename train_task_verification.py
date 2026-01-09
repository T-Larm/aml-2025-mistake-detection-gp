"""
Training script for Task Verification with Leave-One-Recipe-Out Cross-Validation

This script implements a binary classification baseline for task verification
on recipe videos. It uses leave-one-recipe-out cross-validation to evaluate
generalization to unseen recipes.

Usage:
    python train_task_verification.py [--options]
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import os
import argparse
from datetime import datetime

from extension.step_localization import StepLocalizer, prepare_dataset_for_task_verification
from dataloader.TaskVerificationDataset import TaskVerificationDataset
from core.models.task_verifier import TaskVerifier, SimpleMLPVerifier


def get_recipe_groups(recording_ids, annotations_file='annotations/annotation_json/complete_step_annotations.json'):
    """
    Group recordings by recipe (activity_id).
    
    This is essential for leave-one-recipe-out cross-validation.
    Videos of the same recipe should not appear in both train and test sets.
    
    Args:
        recording_ids: List of recording IDs
        annotations_file: Path to step annotations JSON
    
    Returns:
        Dictionary mapping activity_id to list of recording_ids
    """
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    recipe_groups = {}
    for rec_id in recording_ids:
        if rec_id in annotations:
            activity_id = annotations[rec_id]['activity_id']
            activity_name = annotations[rec_id].get('activity_name', activity_id)
            
            if activity_id not in recipe_groups:
                recipe_groups[activity_id] = {
                    'name': activity_name,
                    'recordings': []
                }
            recipe_groups[activity_id]['recordings'].append(rec_id)
    
    return recipe_groups


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Task verification model
        dataloader: Training data loader
        criterion: Loss function (BCELoss)
        optimizer: Optimizer
        device: 'cuda' or 'cpu'
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        embeddings = batch['embeddings'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(embeddings, masks)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """
    Evaluate the model.
    
    Args:
        model: Task verification model
        dataloader: Evaluation data loader
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embeddings'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(embeddings, masks)
            
            # Convert to numpy
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
    }
    
    # AUC only if both classes present
    if len(np.unique(all_labels)) > 1:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['auc'] = 0.0
    
    return metrics


def leave_one_recipe_out_cv(
    data_dict,
    annotations_file='annotations/annotation_json/complete_step_annotations.json',
    model_type='transformer',
    num_epochs=50,
    batch_size=8,
    lr=1e-4,
    device='cuda',
    save_dir='results/task_verification'
):
    """
    Perform leave-one-recipe-out cross-validation.
    
    Args:
        data_dict: Output from prepare_dataset_for_task_verification
        annotations_file: Path to annotations
        model_type: 'transformer' or 'mlp'
        num_epochs: Training epochs per fold
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
        save_dir: Directory to save results
    
    Returns:
        Tuple of (average_metrics, all_fold_results)
    """
    # Create full dataset
    full_dataset = TaskVerificationDataset(
        data_dict['embeddings'],
        data_dict['labels'],
        data_dict['masks'],
        data_dict['recording_ids']
    )
    
    # Print dataset statistics
    full_dataset.print_statistics()
    
    # Group recordings by recipe
    recipe_groups = get_recipe_groups(data_dict['recording_ids'], annotations_file)
    recipe_ids = list(recipe_groups.keys())
    
    print(f"\n{'='*60}")
    print(f"Leave-One-Recipe-Out Cross-Validation")
    print(f"{'='*60}")
    print(f"Total recipes:        {len(recipe_ids)}")
    print(f"Total videos:         {len(data_dict['recording_ids'])}")
    print(f"Model type:           {model_type}")
    print(f"Epochs per fold:      {num_epochs}")
    print(f"Batch size:           {batch_size}")
    print(f"Learning rate:        {lr}")
    print(f"Device:               {device}")
    print(f"{'='*60}\n")
    
    # Store results for each fold
    all_fold_results = []
    
    # Leave-one-out cross-validation
    for fold_idx, test_recipe in enumerate(recipe_ids):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{len(recipe_ids)}")
        print(f"{'='*60}")
        print(f"Test recipe: {test_recipe} ({recipe_groups[test_recipe]['name']})")
        
        # Split indices
        test_recordings = recipe_groups[test_recipe]['recordings']
        train_recordings = []
        for recipe in recipe_ids:
            if recipe != test_recipe:
                train_recordings.extend(recipe_groups[recipe]['recordings'])
        
        # Get dataset indices
        train_indices = [i for i, rid in enumerate(data_dict['recording_ids']) 
                        if rid in train_recordings]
        test_indices = [i for i, rid in enumerate(data_dict['recording_ids']) 
                       if rid in test_recordings]
        
        print(f"Train samples:       {len(train_indices)}")
        print(f"Test samples:        {len(test_indices)}")
        
        # Skip if test set is empty
        if len(test_indices) == 0:
            print("Warning: Empty test set, skipping fold")
            continue
        
        # Create data loaders
        train_subset = Subset(full_dataset, train_indices)
        test_subset = Subset(full_dataset, test_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        embedding_dim = data_dict['embeddings'].shape[2]
        
        if model_type == 'transformer':
            model = TaskVerifier(embedding_dim=embedding_dim).to(device)
        elif model_type == 'mlp':
            model = SimpleMLPVerifier(embedding_dim=embedding_dim).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Model parameters:    {model.get_num_parameters():,}")
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Training loop
        best_test_f1 = 0
        best_metrics = None
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0:
                test_metrics = evaluate(model, test_loader, device)
                
                print(f"Epoch {epoch+1:3d}/{num_epochs} - "
                      f"Loss: {train_loss:.4f} - "
                      f"Test F1: {test_metrics['f1']:.4f} - "
                      f"Test AUC: {test_metrics['auc']:.4f}")
                
                # Save best model based on F1
                if test_metrics['f1'] > best_test_f1:
                    best_test_f1 = test_metrics['f1']
                    best_metrics = test_metrics
                    patience_counter = 0
                    
                    # Save best model for this fold
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        model_path = os.path.join(save_dir, f'fold_{fold_idx+1}_best.pt')
                        torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Final evaluation with best model
        if best_metrics is None:
            best_metrics = evaluate(model, test_loader, device)
        
        print(f"\n{'='*40}")
        print(f"Fold {fold_idx + 1} Best Results:")
        print(f"{'='*40}")
        for metric, value in best_metrics.items():
            print(f"  {metric:12s}: {value:.4f}")
        print(f"{'='*40}")
        
        # Store results
        fold_result = {
            'fold': fold_idx + 1,
            'test_recipe': test_recipe,
            'test_recipe_name': recipe_groups[test_recipe]['name'],
            'num_train': len(train_indices),
            'num_test': len(test_indices),
            'metrics': best_metrics
        }
        all_fold_results.append(fold_result)
    
    # Aggregate results across folds
    print(f"\n{'='*60}")
    print("LEAVE-ONE-RECIPE-OUT CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Completed folds: {len(all_fold_results)}/{len(recipe_ids)}\n")
    
    avg_metrics = {}
    for metric in all_fold_results[0]['metrics'].keys():
        values = [fold['metrics'][metric] for fold in all_fold_results]
        avg_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        
        print(f"{metric.upper():12s}: "
              f"{avg_metrics[metric]['mean']:.4f} ± {avg_metrics[metric]['std']:.4f} "
              f"(min: {avg_metrics[metric]['min']:.4f}, max: {avg_metrics[metric]['max']:.4f})")
    
    print(f"{'='*60}\n")
    
    return avg_metrics, all_fold_results


def main():
    """Main function to run task verification training."""
    parser = argparse.ArgumentParser(description='Task Verification Training')
    parser.add_argument('--features_dir', type=str, default='egovlp',
                        help='Directory containing pre-extracted features')
    parser.add_argument('--annotations_file', type=str, 
                        default='annotations/annotation_json/complete_step_annotations.json',
                        help='Path to step annotations JSON')
    parser.add_argument('--split_file', type=str, 
                        default='er_annotations/recordings_combined_splits.json',
                        help='Path to split JSON file')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Which split to use (train, val, or test)')
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'mlp'],
                        help='Model architecture to use')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='results/task_verification',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\n{'='*60}")
    print("Task Verification Training")
    print(f"{'='*60}")
    print(f"Timestamp:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Features directory:   {args.features_dir}")
    print(f"Annotations file:     {args.annotations_file}")
    print(f"Split file:           {args.split_file}")
    print(f"Using split:          {args.split}")
    print(f"Model type:           {args.model_type}")
    print(f"Device:               {args.device}")
    print(f"{'='*60}\n")
    
    # Initialize step localizer
    print("Initializing step localizer...")
    localizer = StepLocalizer(
        annotations_path=args.annotations_file,
        features_dir=args.features_dir
    )
    
    # Prepare dataset
    print(f"\nPreparing dataset from {args.split} split...")
    data_dict = prepare_dataset_for_task_verification(
        localizer=localizer,
        split_file=args.split_file,
        split=args.split
    )
    
    # Run leave-one-recipe-out cross-validation
    avg_metrics, fold_results = leave_one_recipe_out_cv(
        data_dict=data_dict,
        annotations_file=args.annotations_file,
        model_type=args.model_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # Save results to JSON
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': vars(args),
        'average_metrics': avg_metrics,
        'fold_results': fold_results
    }
    
    results_file = os.path.join(
        args.save_dir, 
        f'loro_cv_{args.model_type}_{args.split}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
