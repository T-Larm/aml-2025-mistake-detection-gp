# Task Verification Baseline - Quick Start Guide

This guide shows you how to use the task verification implementation for Substep 2.

## 📁 Files Created

1. **`core/models/task_verifier.py`** - Model architectures
   - `TaskVerifier`: Transformer-based model (recommended)
   - `SimpleMLPVerifier`: MLP baseline (simpler alternative)

2. **`dataloader/TaskVerificationDataset.py`** - Dataset class
   - Handles step embeddings with variable-length sequences
   - Includes masking for padded sequences
   - Provides dataset statistics

3. **`train_task_verification.py`** - Main training script
   - Leave-one-recipe-out cross-validation
   - Training and evaluation loops
   - Results saving and reporting

## 🚀 Quick Start

### Basic Usage

Run with default settings (transformer model, train split):

```bash
python train_task_verification.py
```

### Advanced Usage

Customize training parameters:

```bash
python train_task_verification.py \
    --features_dir egovlp \
    --split train \
    --model_type transformer \
    --num_epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --device cuda
```

### All Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--features_dir` | `egovlp` | Directory with pre-extracted features |
| `--annotations_file` | `annotations/annotation_json/step_annotations.json` | Path to annotations |
| `--split_file` | `er_annotations/recordings_combined_splits.json` | Path to split file |
| `--split` | `train` | Which split to use (train/val/test) |
| `--model_type` | `transformer` | Model type (transformer/mlp) |
| `--num_epochs` | `50` | Training epochs per fold |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--device` | `cuda` | Device (cuda/cpu) |
| `--save_dir` | `results/task_verification` | Results directory |

## 📊 Expected Output

The script will:

1. **Load data** and extract step embeddings
2. **Print dataset statistics**:
   ```
   ============================================================
   Task Verification Dataset Statistics
   ============================================================
   Total samples:        45
   Positive (errors):    23 (51.1%)
   Negative (no errors): 22 (48.9%)
   
   Embedding dimension:  1024
   Max steps (padded):   12
   Avg steps per video:  7.3
   Min steps per video:  3
   Max steps per video:  12
   ============================================================
   ```

3. **Run leave-one-recipe-out CV**:
   ```
   ============================================================
   Fold 1/10: Testing on recipe pancakes
   ============================================================
   Train samples: 40, Test samples: 5
   Model parameters: 12,345,678
   
   Epoch  10/50 - Loss: 0.6234 - Test F1: 0.6500 - Test AUC: 0.7200
   Epoch  20/50 - Loss: 0.5123 - Test F1: 0.7000 - Test AUC: 0.7800
   ...
   ```

4. **Report final results**:
   ```
   ============================================================
   LEAVE-ONE-RECIPE-OUT CROSS-VALIDATION RESULTS
   ============================================================
   Completed folds: 10/10
   
   ACCURACY    : 0.7234 ± 0.0851 (min: 0.6000, max: 0.8500)
   PRECISION   : 0.7012 ± 0.0923 (min: 0.5500, max: 0.8200)
   RECALL      : 0.6845 ± 0.1034 (min: 0.5000, max: 0.8000)
   F1          : 0.6921 ± 0.0892 (min: 0.5500, max: 0.8100)
   AUC         : 0.7654 ± 0.0712 (min: 0.6500, max: 0.8700)
   ============================================================
   ```

5. **Save results** to JSON file in `results/task_verification/`

## 📈 Understanding the Results

### Key Metrics

- **Accuracy**: Overall correctness (should be > 0.65)
- **Precision**: When predicting "has errors", how often correct
- **Recall**: Of all videos with errors, how many found
- **F1 Score**: Balanced measure (primary metric, should be > 0.60)
- **AUC**: Ability to separate classes (should be > 0.70)

### What's Good Performance?

Given the small dataset and leave-one-recipe-out setting:

- **F1 > 0.60**: Good generalization to new recipes
- **AUC > 0.70**: Model learns meaningful patterns
- **Low std (< 0.10)**: Consistent across recipes

### What if Performance is Low?

If F1 < 0.50 or AUC < 0.60:

1. **Check data balance**: Too few positive/negative samples?
2. **Increase epochs**: Try `--num_epochs 100`
3. **Adjust learning rate**: Try `--lr 5e-5` or `--lr 5e-4`
4. **Try MLP baseline**: `--model_type mlp` (simpler, might work better)
5. **Check features**: Are EgoVLP features properly extracted?

## 🔍 Model Comparison

### Transformer Model (Recommended)

```bash
python train_task_verification.py --model_type transformer
```

**Pros**:
- Captures temporal dependencies between steps
- Attention mechanism learns step interactions
- Better for sequential understanding

**Cons**:
- More parameters (higher overfitting risk)
- Slower training

### MLP Baseline

```bash
python train_task_verification.py --model_type mlp
```

**Pros**:
- Simpler, fewer parameters
- Faster training
- Less prone to overfitting on small data

**Cons**:
- Ignores step ordering
- No attention mechanism
- Might miss temporal patterns

## 📂 Output Files

Results are saved in `results/task_verification/`:

```
results/task_verification/
├── loro_cv_transformer_train_20260108_143052.json  # Full results
├── fold_1_best.pt                                   # Best model weights (fold 1)
├── fold_2_best.pt                                   # Best model weights (fold 2)
└── ...
```

### Results JSON Structure

```json
{
  "timestamp": "2026-01-08 14:30:52",
  "configuration": {
    "model_type": "transformer",
    "num_epochs": 50,
    ...
  },
  "average_metrics": {
    "f1": {
      "mean": 0.6921,
      "std": 0.0892,
      "min": 0.5500,
      "max": 0.8100
    },
    ...
  },
  "fold_results": [
    {
      "fold": 1,
      "test_recipe": "recipe_001",
      "test_recipe_name": "Pancakes",
      "metrics": {...}
    },
    ...
  ]
}
```

## 🐛 Troubleshooting

### Error: "No module named 'extension.step_localization'"

Make sure you have the `extension/step_localization.py` file with the `StepLocalizer` class and `prepare_dataset_for_task_verification` function.

### Error: "CUDA out of memory"

Reduce batch size:
```bash
python train_task_verification.py --batch_size 4
```

Or use CPU:
```bash
python train_task_verification.py --device cpu
```

### Warning: "Empty test set, skipping fold"

Some recipes might not have videos in your split. This is normal if using only train split.

### Low performance (F1 < 0.5)

1. Check if features are properly loaded
2. Verify annotations are correct
3. Try different hyperparameters
4. Consider data augmentation or preprocessing

## 📝 Next Steps

After training:

1. **Analyze per-recipe results**: Which recipes are hardest?
2. **Compare transformer vs MLP**: Which works better?
3. **Error analysis**: Look at misclassified videos
4. **Feature importance**: Which steps contribute most?
5. **Ensemble methods**: Combine multiple models

## 💡 Tips for Better Results

1. **Use both train and val splits**: More data = better generalization
2. **Tune hyperparameters**: Grid search over lr, batch_size, hidden_dim
3. **Add regularization**: Increase dropout if overfitting
4. **Feature engineering**: Try different pooling strategies
5. **Class balancing**: Use weighted loss if imbalanced

---

**Need help?** Check the inline documentation in each file for detailed explanations!
