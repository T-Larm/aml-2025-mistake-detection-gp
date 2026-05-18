import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from extension.step_localization import StepLocalizer, prepare_dataset_for_task_verification
from dataloader.TaskVerificationDataset import TaskVerificationDataset
from core.models.task_verifier import TaskVerifier, SimpleMLPVerifier

def get_recipe_groups(recording_ids, annotations_file='annotations/annotation_json/complete_step_annotations.json'):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    recipe_groups = {}
    for rec_id in recording_ids:
        if rec_id in annotations:
            activity_id = annotations[rec_id]['activity_id']
            activity_name = annotations[rec_id].get('activity_name', activity_id)
            if activity_id not in recipe_groups:
                recipe_groups[activity_id] = {'name': activity_name, 'recordings': []}
            recipe_groups[activity_id]['recordings'].append(rec_id)
    return recipe_groups

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in dataloader:
        embeddings = batch['embeddings'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(embeddings, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embeddings'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(embeddings, masks)
            probs = outputs.cpu().numpy()
            preds = (probs > threshold).astype(int)
            labels_np = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probs.extend(probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
    }
    if len(np.unique(all_labels)) > 1:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['auc'] = 0.0
    return metrics, all_preds, all_labels, all_probs

def load_precomputed_embeddings(npz_path, annotations_file):
    print(f"Loading embeddings from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    all_video_embeddings = []
    all_video_labels = []
    all_video_masks = []
    all_recording_ids = []
    unique_recording_ids = set()
    for key in data.files:
        if key != 'used_hiero' and not key.endswith('_errors'):
            unique_recording_ids.add(key)
    
    max_steps_per_video = 0
    embedding_dim = None
    for rec_id in unique_recording_ids:
        if rec_id in data:
            v_raw = data[rec_id]
            curr_steps = v_raw.shape[0] if v_raw.ndim == 2 else (1 if v_raw.ndim == 1 else 0)
            curr_dim = v_raw.shape[1] if v_raw.ndim == 2 else (v_raw.shape[0] if v_raw.ndim == 1 else 0)
            max_steps_per_video = max(max_steps_per_video, curr_steps)
            if embedding_dim is None and curr_dim > 0:
                embedding_dim = curr_dim
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
        
    for rec_id in tqdm(sorted(list(unique_recording_ids)), desc="Aggregating video data"):
        v_raw = data[rec_id]
        if v_raw.ndim == 2:
            current_embeddings = v_raw
            num_steps = v_raw.shape[0]
        elif v_raw.ndim == 1:
            current_embeddings = v_raw[np.newaxis, :]
            num_steps = 1
        else:
            current_embeddings = np.array([], dtype=np.float32).reshape(0, embedding_dim)
            num_steps = 0
        
        if num_steps > 0:
            scaler = StandardScaler()
            current_embeddings = scaler.fit_transform(current_embeddings)
            
        video_has_errors = False
        if rec_id in annotations:
            steps_data = annotations[rec_id].get('steps', [])
            for step in steps_data:
                if step.get('has_errors'):
                    video_has_errors = True
                    break
                    
        padded_embeddings = np.zeros((max_steps_per_video, embedding_dim), dtype=np.float32)
        mask = np.zeros(max_steps_per_video, dtype=bool)
        if num_steps > 0:
            padded_embeddings[:num_steps, :] = current_embeddings
            mask[:num_steps] = True
            
        all_video_embeddings.append(padded_embeddings)
        all_video_labels.append(int(video_has_errors))
        all_video_masks.append(mask)
        all_recording_ids.append(rec_id)
        
    return {
        'embeddings': np.array(all_video_embeddings, dtype=np.float32),
        'labels': np.array(all_video_labels, dtype=int),
        'masks': np.array(all_video_masks, dtype=bool),
        'recording_ids': all_recording_ids,
        'used_hiero': data['used_hiero'].item() if 'used_hiero' in data else False
    }

def main():
    parser = argparse.ArgumentParser(description='Task Verification Training (Colab Logic)')
    parser.add_argument('--precomputed_embeddings', type=str, default=None,
                        help='Path to precomputed embeddings NPZ to use instead of features_dir')
    parser.add_argument('--features_dir', type=str, default='egovlp',
                        help='Directory containing pre-extracted features (if not using precomputed)')
    parser.add_argument('--annotations_file', type=str, 
                        default='annotations/annotation_json/complete_step_annotations.json',
                        help='Path to step annotations JSON')
    parser.add_argument('--split_file', type=str, 
                        default='er_annotations/recordings_combined_splits.json',
                        help='Path to split JSON file (if using StepLocalizer)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'mlp'])
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--threshold', type=float, default=0.57)
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--enable_early_stopping', action='store_true', default=True, help='Enable early stopping based on patience')
    parser.add_argument('--average_last_n_evals', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='results/task_verification')
    
    args = parser.parse_args()
    
    if args.precomputed_embeddings:
        data_dict = load_precomputed_embeddings(args.precomputed_embeddings, args.annotations_file)
    else:
        localizer = StepLocalizer(annotations_path=args.annotations_file, features_dir=args.features_dir)
        data_dict = prepare_dataset_for_task_verification(localizer=localizer, split_file=args.split_file, split=args.split)
        
    full_dataset = TaskVerificationDataset(data_dict['embeddings'], data_dict['labels'], data_dict['masks'], data_dict['recording_ids'])
    full_dataset.print_statistics()
    
    recipe_groups = get_recipe_groups(data_dict['recording_ids'], args.annotations_file)
    recipe_ids = list(recipe_groups.keys())
    
    all_fold_results = []
    fold_predictions = {}
    
    for fold_idx, test_recipe in enumerate(recipe_ids):
        print(f"\nFold {fold_idx + 1}/{len(recipe_ids)} - Test recipe: {test_recipe}")
        test_recordings = recipe_groups[test_recipe]['recordings']
        train_recordings = []
        for r in recipe_ids:
            if r != test_recipe:
                train_recordings.extend(recipe_groups[r]['recordings'])
                
        train_indices = [i for i, rid in enumerate(data_dict['recording_ids']) if rid in train_recordings]
        test_indices = [i for i, rid in enumerate(data_dict['recording_ids']) if rid in test_recordings]
        
        if len(test_indices) == 0:
            continue
            
        train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=args.batch_size, shuffle=False)
        
        em_dim = data_dict['embeddings'].shape[2]
        
        if args.model_type == 'transformer':
            model = TaskVerifier(embedding_dim=em_dim, hidden_dim=args.hidden_dim, num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout).to(args.device)
        else:
            model = SimpleMLPVerifier(embedding_dim=em_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(args.device)
            
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_test_f1 = 0
        best_metrics_single_epoch = None
        best_preds, best_labels, best_probs = [], [], []
        patience_counter = 0
        training_history = {'loss': [], 'train_accuracy': [], 'test_accuracy': [], 'test_f1': [], 'test_auc': [], 'test_precision': [], 'test_recall': []}
        
        for epoch in range(args.num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
            training_history['loss'].append(train_loss)
            
            if (epoch + 1) % 5 == 0:
                train_metrics, _, _, _ = evaluate(model, train_loader, args.device, args.threshold)
                test_metrics, preds, labels, probs = evaluate(model, test_loader, args.device, args.threshold)
                
                for m, h in [('accuracy', 'train_accuracy')]: training_history[h].append(train_metrics[m])
                for m, h in [('accuracy', 'test_accuracy'), ('f1', 'test_f1'), ('auc', 'test_auc'), ('precision', 'test_precision'), ('recall', 'test_recall')]: training_history[h].append(test_metrics[m])
                
                if test_metrics['f1'] > best_test_f1:
                    best_test_f1 = test_metrics['f1']
                    best_metrics_single_epoch = test_metrics
                    best_preds, best_labels, best_probs = preds, labels, probs
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if args.enable_early_stopping and patience_counter >= args.patience:
                    break
                    
        num_to_average = min(args.average_last_n_evals, len(training_history['test_f1']))
        if num_to_average == 0 or best_metrics_single_epoch is None:
            final_metrics = best_metrics_single_epoch if best_metrics_single_epoch is not None else {'accuracy':0.0, 'precision':0.0, 'recall':0.0, 'f1':0.0, 'auc':0.0}
            final_preds, final_labels, final_probs = best_preds, best_labels, best_probs
        else:
            final_metrics = {m: np.mean(training_history[f'test_{m}'][-num_to_average:]) for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
            final_preds, final_labels, final_probs = preds, labels, probs
            
        all_fold_results.append({
            'fold': fold_idx + 1, 'test_recipe': test_recipe, 'test_recipe_name': recipe_groups[test_recipe]['name'],
            'num_train': len(train_indices), 'num_test': len(test_indices), 'metrics': final_metrics, 'training_history': training_history
        })
        fold_predictions[test_recipe] = {'predictions': final_preds, 'labels': final_labels, 'probabilities': final_probs, 'recording_ids': test_recordings}
        
    # Aggregate
    avg_metrics = {}
    for metric in all_fold_results[0]['metrics'].keys():
        values = [fold['metrics'][metric] for fold in all_fold_results]
        avg_metrics[metric] = {'mean': np.mean(values), 'std': np.std(values), 'min': np.min(values), 'max': np.max(values), 'values': values}

    # Plot & Save logic
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Boxplot
    ax = axes[0, 0]
    m_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    bp = ax.boxplot([avg_metrics[m]['values'] for m in m_plot], labels=[m.upper() for m in m_plot], patch_artist=True, showmeans=True)
    for patch in bp['boxes']: patch.set_facecolor('#3498db'); patch.set_alpha(0.6)
    ax.set_title('Metrics Distribution Across Folds'); ax.set_ylim([0, 1])
    
    # 2. Per-recipe
    ax = axes[0, 1]
    r_names = [f['test_recipe_name'][:20] for f in all_fold_results]
    f1s = [f['metrics']['f1'] for f in all_fold_results]
    colors = ['#e74c3c' if f < 0.5 else '#f39c12' if f < 0.7 else '#2ecc71' for f in f1s]
    ax.barh(r_names, f1s, color=colors, alpha=0.7)
    ax.axvline(avg_metrics['f1']['mean'], color='blue', linestyle='--')
    ax.set_title('F1 Score per Recipe')
    
    # 3. Curve
    ax = axes[1, 0]
    if all_fold_results:
        hist = all_fold_results[0]['training_history']
        ax.plot(range(1, len(hist['loss']) + 1), hist['loss'], 'b-', label='Training Loss')
        ax2 = ax.twinx()
        ax2.plot(list(range(5, len(hist['loss']) + 1, 5)), hist['test_f1'], 'r-o', label='Test F1')
        ax.set_title("Training Curve (Fold 1)")
    
    # 4. Confusion matrix
    ax = axes[1, 1]
    all_preds_agg, all_labels_agg = [], []
    for r, pd_ in fold_predictions.items():
        all_preds_agg.extend(pd_['predictions'])
        all_labels_agg.extend(pd_['labels'])
    sns.heatmap(confusion_matrix(all_labels_agg, all_preds_agg), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Aggregated Confusion Matrix')
    
    os.makedirs(args.save_dir, exist_ok=True)
    fig_path = os.path.join(args.save_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(fig_path, bbox_inches='tight')
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': vars(args),
        'average_metrics': {k: {key: val for key, val in v.items() if key != 'values'} for k, v in avg_metrics.items()},
        'fold_results': all_fold_results
    }
    res_file = os.path.join(args.save_dir, f"loro_cv_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(res_file, 'w') as f: json.dump(results, f, indent=2)
    
    df = pd.DataFrame([{**{'fold': f['fold'], 'test_recipe_name': f['test_recipe_name']}, **{f'metric_{m}': v for m, v in f['metrics'].items()}} for f in all_fold_results])
    df = pd.concat([df, pd.DataFrame([{'test_recipe_name': 'Average', **{f'metric_{m}': v['mean'] for m, v in avg_metrics.items()}}])], ignore_index=True)
    df.to_csv(res_file.replace('.json', '.csv'), index=False)
    print(f"Results saved to {args.save_dir}")

if __name__ == '__main__':
    main()