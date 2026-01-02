"""
Extension Substep 1: Recipe Step Localization

This module extracts step-level embeddings from video features using:
- Route A: GT (Ground Truth) boundaries - as an upper bound baseline
- Route B: Predicted boundaries (e.g., from ActionFormer) - for end-to-end evaluation

For each video, it:
1. Loads step boundaries (GT or predicted)
2. Loads the pre-extracted EgoVLP features
3. Computes step-level embeddings by averaging frame features within each step boundary

Output: A dictionary containing step embeddings for each video, ready for subsequent tasks.
"""

import json
import os
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field


@dataclass
class PredictedStep:
    """A predicted step from ActionFormer or similar model."""
    start_time: float
    end_time: float
    confidence: float
    label: Optional[int] = None  # Predicted action label if available


@dataclass
class StepInfo:
    """Information about a single step in a video."""
    step_id: int
    start_time: float
    end_time: float
    description: str
    has_errors: bool
    embedding: Optional[np.ndarray] = None


@dataclass
class VideoStepData:
    """All step data for a single video."""
    recording_id: str
    activity_id: int
    activity_name: str
    steps: List[StepInfo]
    video_label: int  # 0 = all correct, 1 = has errors


class StepLocalizer:
    """
    Extracts step-level embeddings from video features using GT or predicted boundaries.
    """
    
    def __init__(
        self,
        annotations_path: str,
        features_dir: str,
        fps: float = 1.0,
        feature_key: str = 'arr_0'
    ):
        """
        Args:
            annotations_path: Path to complete_step_annotations.json
            features_dir: Directory containing EgoVLP .npz feature files
            fps: Frame rate of the extracted features (default: 1 FPS)
            feature_key: Key to access features in .npz file (default: 'arr_0')
        """
        self.annotations_path = annotations_path
        self.features_dir = features_dir
        self.fps = fps
        self.feature_key = feature_key
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        print(f"Loaded annotations for {len(self.annotations)} videos")
    
    def _get_feature_filename(self, recording_id: str) -> str:
        """
        Convert recording_id to feature filename.
        Example: "9_8" -> "9_8_360p_224.mp4_1s_1s.npz"
        """
        return f"{recording_id}_360p_224.mp4_1s_1s.npz"
    
    def _load_features(self, recording_id: str) -> Optional[np.ndarray]:
        """Load EgoVLP features for a video."""
        filename = self._get_feature_filename(recording_id)
        filepath = os.path.join(self.features_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: Feature file not found: {filepath}")
            return None
        
        data = np.load(filepath)
        features = data[self.feature_key].astype(np.float32)  # Convert from float16
        return features
    
    def _time_to_frame(self, time_sec: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time_sec * self.fps)
    
    def _extract_step_embedding(
        self,
        features: np.ndarray,
        start_time: float,
        end_time: float
    ) -> np.ndarray:
        """
        Extract step embedding by averaging frame features within the time boundary.
        
        Args:
            features: Video features of shape (T, D)
            start_time: Step start time in seconds
            end_time: Step end time in seconds
        
        Returns:
            Step embedding of shape (D,)
        """
        start_frame = self._time_to_frame(start_time)
        end_frame = self._time_to_frame(end_time)
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(len(features), end_frame + 1)  # +1 because end is inclusive
        
        if start_frame >= end_frame:
            # Edge case: very short step, just take the nearest frame
            frame_idx = min(start_frame, len(features) - 1)
            return features[frame_idx]
        
        # Mean pooling over the step duration
        step_features = features[start_frame:end_frame]
        step_embedding = np.mean(step_features, axis=0)
        
        return step_embedding
    
    def process_video(self, recording_id: str) -> Optional[VideoStepData]:
        """
        Process a single video: extract step embeddings using GT boundaries.
        
        Args:
            recording_id: Video ID (e.g., "9_8")
        
        Returns:
            VideoStepData object containing all step information and embeddings
        """
        if recording_id not in self.annotations:
            print(f"Warning: No annotations found for {recording_id}")
            return None
        
        # Load features
        features = self._load_features(recording_id)
        if features is None:
            return None
        
        # Get annotation
        ann = self.annotations[recording_id]
        
        # Process each step
        steps = []
        has_any_error = False
        
        for step_ann in ann['steps']:
            step_info = StepInfo(
                step_id=step_ann['step_id'],
                start_time=step_ann['start_time'],
                end_time=step_ann['end_time'],
                description=step_ann['description'],
                has_errors=step_ann['has_errors']
            )
            
            # Extract embedding
            step_info.embedding = self._extract_step_embedding(
                features,
                step_info.start_time,
                step_info.end_time
            )
            
            steps.append(step_info)
            
            if step_info.has_errors:
                has_any_error = True
        
        # Sort steps by start time (they might not be in order)
        steps.sort(key=lambda s: s.start_time)
        
        video_data = VideoStepData(
            recording_id=recording_id,
            activity_id=ann['activity_id'],
            activity_name=ann['activity_name'],
            steps=steps,
            video_label=1 if has_any_error else 0
        )
        
        return video_data
    
    def process_all_videos(
        self,
        recording_ids: Optional[List[str]] = None
    ) -> Dict[str, VideoStepData]:
        """
        Process multiple videos.
        
        Args:
            recording_ids: List of video IDs to process. If None, process all available.
        
        Returns:
            Dictionary mapping recording_id to VideoStepData
        """
        if recording_ids is None:
            recording_ids = list(self.annotations.keys())
        
        results = {}
        success_count = 0
        
        for recording_id in recording_ids:
            video_data = self.process_video(recording_id)
            if video_data is not None:
                results[recording_id] = video_data
                success_count += 1
        
        print(f"Successfully processed {success_count}/{len(recording_ids)} videos")
        return results
    
    def get_step_embeddings_matrix(
        self,
        video_data: VideoStepData,
        pad_to_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Get step embeddings as a matrix for a video.
        
        Args:
            video_data: Processed video data
            pad_to_length: If specified, pad/truncate to this length
        
        Returns:
            - embeddings: (num_steps, embedding_dim) or (pad_to_length, embedding_dim)
            - mask: (num_steps,) or (pad_to_length,) boolean mask for valid steps
            - actual_length: actual number of steps
        """
        embeddings = np.stack([s.embedding for s in video_data.steps], axis=0)
        num_steps = len(video_data.steps)
        
        if pad_to_length is None:
            mask = np.ones(num_steps, dtype=bool)
            return embeddings, mask, num_steps
        
        # Pad or truncate
        embed_dim = embeddings.shape[1]
        padded = np.zeros((pad_to_length, embed_dim), dtype=np.float32)
        mask = np.zeros(pad_to_length, dtype=bool)
        
        actual_len = min(num_steps, pad_to_length)
        padded[:actual_len] = embeddings[:actual_len]
        mask[:actual_len] = True
        
        return padded, mask, actual_len


def prepare_dataset_for_task_verification(
    localizer: StepLocalizer,
    split_file: str,
    split: str = 'train'
) -> Dict:
    """
    Prepare step embeddings for task verification (Extension Substep 2).
    
    Args:
        localizer: StepLocalizer instance
        split_file: Path to split JSON file (e.g., recordings_combined_splits.json)
        split: 'train', 'val', or 'test'
    
    Returns:
        Dictionary with 'embeddings', 'labels', 'masks', 'recording_ids'
    """
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    recording_ids = splits[split]
    print(f"Processing {len(recording_ids)} videos for {split} split...")
    
    # Process all videos
    video_data_dict = localizer.process_all_videos(recording_ids)
    
    # Find max number of steps for padding
    max_steps = max(len(vd.steps) for vd in video_data_dict.values())
    print(f"Max steps in a video: {max_steps}")
    
    # Prepare arrays
    all_embeddings = []
    all_labels = []
    all_masks = []
    valid_ids = []
    
    for recording_id in recording_ids:
        if recording_id not in video_data_dict:
            continue
        
        video_data = video_data_dict[recording_id]
        embeddings, mask, _ = localizer.get_step_embeddings_matrix(
            video_data,
            pad_to_length=max_steps
        )
        
        all_embeddings.append(embeddings)
        all_labels.append(video_data.video_label)
        all_masks.append(mask)
        valid_ids.append(recording_id)
    
    result = {
        'embeddings': np.stack(all_embeddings, axis=0),  # (N, max_steps, embed_dim)
        'labels': np.array(all_labels),  # (N,)
        'masks': np.stack(all_masks, axis=0),  # (N, max_steps)
        'recording_ids': valid_ids,
        'max_steps': max_steps
    }
    
    print(f"Dataset shape: {result['embeddings'].shape}")
    print(f"Positive samples (has errors): {sum(all_labels)}/{len(all_labels)}")
    
    return result


# ==============================================================================
# ROUTE B: Predicted Boundaries (ActionFormer or other models)
# ==============================================================================

class PredictedBoundaryLocalizer:
    """
    Route B: Extracts step-level embeddings using PREDICTED boundaries.
    
    This is for evaluating the end-to-end system where step boundaries
    come from a temporal action detection model (e.g., ActionFormer).
    """
    
    def __init__(
        self,
        features_dir: str,
        predictions_path: Optional[str] = None,
        fps: float = 1.0,
        feature_key: str = 'arr_0',
        confidence_threshold: float = 0.0,
        nms_threshold: float = 0.5,
        max_predictions: Optional[int] = None
    ):
        """
        Args:
            features_dir: Directory containing EgoVLP .npz feature files
            predictions_path: Path to JSON file with ActionFormer predictions
            fps: Frame rate of the extracted features (default: 1 FPS)
            feature_key: Key to access features in .npz file
            confidence_threshold: Minimum confidence to keep a prediction
            nms_threshold: IoU threshold for NMS filtering
            max_predictions: Maximum number of predictions to keep per video
        """
        self.features_dir = features_dir
        self.fps = fps
        self.feature_key = feature_key
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_predictions = max_predictions
        
        # Load predictions if provided
        self.predictions = {}
        if predictions_path and os.path.exists(predictions_path):
            with open(predictions_path, 'r') as f:
                self.predictions = json.load(f)
            print(f"Loaded predictions for {len(self.predictions)} videos")
    
    def load_predictions_from_file(self, predictions_path: str):
        """Load predictions from a JSON file."""
        with open(predictions_path, 'r') as f:
            self.predictions = json.load(f)
        print(f"Loaded predictions for {len(self.predictions)} videos")
    
    def set_predictions(self, predictions: Dict[str, List[Dict]]):
        """
        Set predictions directly from a dictionary.
        
        Expected format:
        {
            "recording_id": [
                {"start": 10.5, "end": 25.3, "confidence": 0.85, "label": 1},
                ...
            ]
        }
        """
        self.predictions = predictions
    
    def _compute_iou(self, seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
        """Compute IoU between two temporal segments."""
        start1, end1 = seg1
        start2, end2 = seg2
        
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)
        
        if union <= 0:
            return 0.0
        return intersection / union
    
    def _nms(self, predictions: List[PredictedStep]) -> List[PredictedStep]:
        """
        Apply Non-Maximum Suppression to filter overlapping predictions.
        
        Args:
            predictions: List of PredictedStep objects, sorted by confidence (desc)
        
        Returns:
            Filtered list of predictions
        """
        if len(predictions) == 0:
            return []
        
        # Sort by confidence (descending)
        sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while sorted_preds:
            # Keep the highest confidence prediction
            best = sorted_preds.pop(0)
            keep.append(best)
            
            # Remove predictions with high IoU overlap
            sorted_preds = [
                p for p in sorted_preds
                if self._compute_iou(
                    (best.start_time, best.end_time),
                    (p.start_time, p.end_time)
                ) < self.nms_threshold
            ]
        
        return keep
    
    def _filter_predictions(
        self,
        predictions: List[PredictedStep]
    ) -> List[PredictedStep]:
        """
        Apply confidence thresholding, NMS, and max predictions limit.
        """
        # 1. Confidence threshold
        filtered = [p for p in predictions if p.confidence >= self.confidence_threshold]
        
        # 2. NMS
        filtered = self._nms(filtered)
        
        # 3. Limit number of predictions
        if self.max_predictions is not None:
            filtered = filtered[:self.max_predictions]
        
        # 4. Sort by start time
        filtered.sort(key=lambda x: x.start_time)
        
        return filtered
    
    def _get_feature_filename(self, recording_id: str) -> str:
        """Convert recording_id to feature filename."""
        return f"{recording_id}_360p_224.mp4_1s_1s.npz"
    
    def _load_features(self, recording_id: str) -> Optional[np.ndarray]:
        """Load EgoVLP features for a video."""
        filename = self._get_feature_filename(recording_id)
        filepath = os.path.join(self.features_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: Feature file not found: {filepath}")
            return None
        
        data = np.load(filepath)
        features = data[self.feature_key].astype(np.float32)
        return features
    
    def _time_to_frame(self, time_sec: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time_sec * self.fps)
    
    def _extract_step_embedding(
        self,
        features: np.ndarray,
        start_time: float,
        end_time: float
    ) -> np.ndarray:
        """Extract step embedding by averaging frame features."""
        start_frame = self._time_to_frame(start_time)
        end_frame = self._time_to_frame(end_time)
        
        start_frame = max(0, start_frame)
        end_frame = min(len(features), end_frame + 1)
        
        if start_frame >= end_frame:
            frame_idx = min(start_frame, len(features) - 1)
            return features[frame_idx]
        
        step_features = features[start_frame:end_frame]
        return np.mean(step_features, axis=0)
    
    def process_video(
        self,
        recording_id: str,
        video_label: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Process a single video using predicted boundaries.
        
        Args:
            recording_id: Video ID
            video_label: Optional ground truth label (0=correct, 1=has errors)
        
        Returns:
            Dictionary with step embeddings and metadata
        """
        if recording_id not in self.predictions:
            print(f"Warning: No predictions found for {recording_id}")
            return None
        
        features = self._load_features(recording_id)
        if features is None:
            return None
        
        # Parse predictions
        raw_preds = self.predictions[recording_id]
        predictions = [
            PredictedStep(
                start_time=p.get('start', p.get('start_time', 0)),
                end_time=p.get('end', p.get('end_time', 0)),
                confidence=p.get('confidence', p.get('score', 1.0)),
                label=p.get('label', None)
            )
            for p in raw_preds
        ]
        
        # Filter predictions
        filtered_preds = self._filter_predictions(predictions)
        
        if len(filtered_preds) == 0:
            print(f"Warning: No predictions remaining after filtering for {recording_id}")
            return None
        
        # Extract embeddings
        step_embeddings = []
        step_info = []
        
        for pred in filtered_preds:
            embedding = self._extract_step_embedding(
                features, pred.start_time, pred.end_time
            )
            step_embeddings.append(embedding)
            step_info.append({
                'start_time': pred.start_time,
                'end_time': pred.end_time,
                'confidence': pred.confidence,
                'label': pred.label
            })
        
        return {
            'recording_id': recording_id,
            'embeddings': np.stack(step_embeddings, axis=0),  # (num_steps, embed_dim)
            'step_info': step_info,
            'num_steps': len(step_embeddings),
            'video_label': video_label
        }
    
    def process_all_videos(
        self,
        recording_ids: Optional[List[str]] = None,
        labels: Optional[Dict[str, int]] = None
    ) -> Dict[str, Dict]:
        """Process multiple videos."""
        if recording_ids is None:
            recording_ids = list(self.predictions.keys())
        
        results = {}
        for recording_id in recording_ids:
            label = labels.get(recording_id) if labels else None
            result = self.process_video(recording_id, video_label=label)
            if result is not None:
                results[recording_id] = result
        
        print(f"Successfully processed {len(results)}/{len(recording_ids)} videos")
        return results


def compare_gt_vs_predicted(
    gt_localizer: 'StepLocalizer',
    pred_localizer: PredictedBoundaryLocalizer,
    recording_ids: List[str]
) -> Dict:
    """
    Compare step localization results between GT and predicted boundaries.
    
    This helps analyze how much performance loss comes from the segmentation
    step vs. the downstream task verification model.
    """
    comparison = {
        'recording_ids': [],
        'gt_num_steps': [],
        'pred_num_steps': [],
        'gt_total_duration': [],
        'pred_total_duration': []
    }
    
    for recording_id in recording_ids:
        gt_data = gt_localizer.process_video(recording_id)
        pred_data = pred_localizer.process_video(recording_id)
        
        if gt_data is None or pred_data is None:
            continue
        
        comparison['recording_ids'].append(recording_id)
        comparison['gt_num_steps'].append(len(gt_data.steps))
        comparison['pred_num_steps'].append(pred_data['num_steps'])
        
        gt_duration = sum(s.end_time - s.start_time for s in gt_data.steps)
        pred_duration = sum(
            s['end_time'] - s['start_time'] for s in pred_data['step_info']
        )
        comparison['gt_total_duration'].append(gt_duration)
        comparison['pred_total_duration'].append(pred_duration)
    
    # Summary statistics
    print("\n=== GT vs Predicted Comparison ===")
    print(f"Videos compared: {len(comparison['recording_ids'])}")
    print(f"Avg GT steps: {np.mean(comparison['gt_num_steps']):.1f}")
    print(f"Avg Predicted steps: {np.mean(comparison['pred_num_steps']):.1f}")
    
    return comparison


# ============== Example Usage ==============
if __name__ == "__main__":
    # Paths (adjust for your setup)
    ANNOTATIONS_PATH = "annotations/complete_step_annotations.json"
    FEATURES_DIR = "egovlp"  # Directory containing .npz files
    SPLIT_FILE = "er_annotations/recordings_combined_splits.json"
    
    # ==========================================
    # ROUTE A: Using Ground Truth Boundaries
    # ==========================================
    print("\n" + "="*60)
    print("ROUTE A: Ground Truth Boundaries")
    print("="*60)
    
    gt_localizer = StepLocalizer(
        annotations_path=ANNOTATIONS_PATH,
        features_dir=FEATURES_DIR,
        fps=1.0
    )
    
    # Example: Process a single video with GT boundaries
    print("\n=== Processing single video (GT) ===")
    video_data = gt_localizer.process_video("9_8")
    if video_data:
        print(f"Video: {video_data.recording_id}")
        print(f"Activity: {video_data.activity_name}")
        print(f"Number of steps: {len(video_data.steps)}")
        print(f"Video label (0=correct, 1=has errors): {video_data.video_label}")
        print("\nSteps:")
        for step in video_data.steps:
            print(f"  Step {step.step_id}: {step.start_time:.1f}s - {step.end_time:.1f}s")
            print(f"    Description: {step.description[:50]}...")
            print(f"    Has errors: {step.has_errors}")
            print(f"    Embedding shape: {step.embedding.shape}")
    
    # ==========================================
    # ROUTE B: Using Predicted Boundaries
    # ==========================================
    print("\n" + "="*60)
    print("ROUTE B: Predicted Boundaries (ActionFormer)")
    print("="*60)
    
    # Example: Create mock predictions for demonstration
    # In practice, load from ActionFormer output
    mock_predictions = {
        "9_8": [
            {"start": 0, "end": 70, "confidence": 0.8},
            {"start": 72, "end": 96, "confidence": 0.75},
            {"start": 110, "end": 175, "confidence": 0.7},
            {"start": 180, "end": 225, "confidence": 0.65},
            {"start": 235, "end": 290, "confidence": 0.6},
            {"start": 300, "end": 375, "confidence": 0.55},
            {"start": 370, "end": 420, "confidence": 0.5},
            {"start": 410, "end": 525, "confidence": 0.45},
            {"start": 520, "end": 560, "confidence": 0.4},
        ]
    }
    
    pred_localizer = PredictedBoundaryLocalizer(
        features_dir=FEATURES_DIR,
        fps=1.0,
        confidence_threshold=0.3,  # Filter low confidence
        nms_threshold=0.5,         # Remove overlapping predictions
        max_predictions=20         # Limit number of steps
    )
    pred_localizer.set_predictions(mock_predictions)
    
    print("\n=== Processing single video (Predicted) ===")
    pred_data = pred_localizer.process_video("9_8", video_label=1)
    if pred_data:
        print(f"Video: {pred_data['recording_id']}")
        print(f"Number of predicted steps: {pred_data['num_steps']}")
        print(f"Embeddings shape: {pred_data['embeddings'].shape}")
        print("\nPredicted steps:")
        for i, step in enumerate(pred_data['step_info']):
            print(f"  Step {i+1}: {step['start_time']:.1f}s - {step['end_time']:.1f}s (conf: {step['confidence']:.2f})")
    
    # Example: Prepare full dataset for task verification
    # print("\n=== Preparing train dataset ===")
    # train_data = prepare_dataset_for_task_verification(
    #     gt_localizer, SPLIT_FILE, split='train'
    # )

