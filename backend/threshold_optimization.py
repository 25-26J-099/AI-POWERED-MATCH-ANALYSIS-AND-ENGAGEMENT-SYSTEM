# threshold_optimization.py
import numpy as np
import json
from itertools import product
from collections import defaultdict

class ThresholdOptimizer:
    """Grid search for optimal Re-ID thresholds."""
    
    def __init__(self, ground_truth_path):
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
    
    def evaluate_thresholds(self, predictions, appearance_thresh, spatial_thresh, combined_thresh):
        """
        Evaluate ID assignment accuracy for given thresholds.
        
        predictions: {frame_idx: {track_id: predicted_id}}
        Returns: (accuracy, num_switches, avg_switch_duration)
        """
        correct = 0
        total = 0
        switches = defaultdict(list)  # {gt_id: [frame_idxs where switch occurred]}
        
        for frame_idx_str, gt_ids in self.ground_truth.items():
            frame_idx = int(frame_idx_str)
            if frame_idx not in predictions:
                continue
            
            pred_ids = predictions[frame_idx]
            
            for track_id_str, gt_id in gt_ids.items():
                track_id = int(track_id_str)
                if track_id not in pred_ids:
                    continue
                
                pred_id = pred_ids[track_id]
                
                if pred_id == gt_id:
                    correct += 1
                else:
                    switches[gt_id].append(frame_idx)
                
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        num_switches = sum(len(frames) for frames in switches.values())
        
        # Compute average switch duration
        switch_durations = []
        for frames in switches.values():
            if len(frames) > 1:
                durations = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
                switch_durations.extend(durations)
        
        avg_duration = np.mean(switch_durations) if switch_durations else 0.0
        
        return accuracy, num_switches, avg_duration
    
    def grid_search(self, prediction_function, 
                   appearance_range=(0.40, 0.45, 0.50, 0.55, 0.60),
                   spatial_range=(150, 200, 250, 300, 350),
                   combined_range=(0.60, 0.65, 0.70, 0.75)):
        """
        Grid search over threshold parameters.
        
        prediction_function: callable that takes thresholds and returns predictions
        """
        results = []
        
        for app_t, spa_t, com_t in product(appearance_range, spatial_range, combined_range):
            print(f"Testing: app={app_t}, spa={spa_t}, com={com_t}")
            
            # Run prediction with these thresholds
            predictions = prediction_function(app_t, spa_t, com_t)
            
            # Evaluate
            accuracy, switches, avg_dur = self.evaluate_thresholds(
                predictions, app_t, spa_t, com_t
            )
            
            results.append({
                'appearance_threshold': app_t,
                'spatial_threshold': spa_t,
                'combined_threshold': com_t,
                'accuracy': accuracy,
                'num_switches': switches,
                'avg_switch_duration': avg_dur,
                'score': accuracy - 0.01 * switches  # Combined metric
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results

# Usage:
# optimizer = ThresholdOptimizer('ground_truth.json')
# 
# def run_pipeline_with_thresholds(app_t, spa_t, com_t):
#     # Modify config
#     config.reid.appearance_threshold = app_t
#     config.reid.spatial_threshold = spa_t
#     # ... run pipeline
#     return predictions
# 
# results = optimizer.grid_search(run_pipeline_with_thresholds)
# print("Best thresholds:", results[0])