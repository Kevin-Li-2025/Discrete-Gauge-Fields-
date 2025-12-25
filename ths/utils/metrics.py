"""
Evaluation metrics for drift detection.

Provides standard metrics:
- Detection delay: samples between true and detected drift
- False positive rate: incorrect detections
- True positive rate: correct detections
- F1 score: harmonic mean of precision and recall
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional


def detection_delay(
    true_drift_points: List[int],
    detected_drift_points: List[int],
    tolerance: int = 50
) -> Tuple[float, List[int]]:
    """
    Compute average detection delay.
    
    For each true drift point, find the nearest detection within
    tolerance and compute the delay.
    
    Args:
        true_drift_points: Indices where true drift occurred
        detected_drift_points: Indices where drift was detected
        tolerance: Maximum distance to count as detected
        
    Returns:
        Tuple of (average delay, list of individual delays)
    """
    if not true_drift_points or not detected_drift_points:
        return float('inf'), []
    
    delays = []
    
    for true_point in true_drift_points:
        # Find detections after true drift
        future_detections = [d for d in detected_drift_points if d >= true_point]
        
        if not future_detections:
            delays.append(tolerance)  # Missed
        else:
            delay = min(future_detections) - true_point
            if delay <= tolerance:
                delays.append(delay)
            else:
                delays.append(tolerance)  # Too late = missed
    
    return np.mean(delays), delays


def false_positive_rate(
    true_drift_points: List[int],
    detected_drift_points: List[int],
    total_samples: int,
    tolerance: int = 50
) -> float:
    """
    Compute false positive rate.
    
    A detection is a false positive if it's not within tolerance
    of any true drift point.
    
    Args:
        true_drift_points: True drift indices
        detected_drift_points: Detected drift indices
        total_samples: Total number of samples in stream
        tolerance: Window around true drift to count as correct
        
    Returns:
        FPR = false_positives / (total_samples - true_drift_windows)
    """
    if not detected_drift_points:
        return 0.0
    
    # Count false positives
    false_positives = 0
    for detection in detected_drift_points:
        is_near_true = any(
            abs(detection - true_point) <= tolerance
            for true_point in true_drift_points
        )
        if not is_near_true:
            false_positives += 1
    
    # Calculate negatives (samples not near any drift)
    drift_window_size = tolerance * 2 * len(true_drift_points)
    total_negatives = max(1, total_samples - drift_window_size)
    
    return false_positives / total_negatives


def true_positive_rate(
    true_drift_points: List[int],
    detected_drift_points: List[int],
    tolerance: int = 50
) -> float:
    """
    Compute true positive rate (recall).
    
    Fraction of true drifts that were detected.
    
    Args:
        true_drift_points: True drift indices
        detected_drift_points: Detected drift indices
        tolerance: Window to count as detected
        
    Returns:
        TPR = detected_true_drifts / total_true_drifts
    """
    if not true_drift_points:
        return 1.0  # No drifts to detect
    
    detected_count = 0
    for true_point in true_drift_points:
        is_detected = any(
            abs(detection - true_point) <= tolerance
            for detection in detected_drift_points
        )
        if is_detected:
            detected_count += 1
    
    return detected_count / len(true_drift_points)


def precision(
    true_drift_points: List[int],
    detected_drift_points: List[int],
    tolerance: int = 50
) -> float:
    """
    Compute precision.
    
    Fraction of detections that correspond to true drifts.
    """
    if not detected_drift_points:
        return 1.0  # No false positives
    
    true_positives = 0
    for detection in detected_drift_points:
        is_near_true = any(
            abs(detection - true_point) <= tolerance
            for true_point in true_drift_points
        )
        if is_near_true:
            true_positives += 1
    
    return true_positives / len(detected_drift_points)


def f1_score(
    true_drift_points: List[int],
    detected_drift_points: List[int],
    tolerance: int = 50
) -> float:
    """
    Compute F1 score.
    
    Harmonic mean of precision and recall.
    """
    p = precision(true_drift_points, detected_drift_points, tolerance)
    r = true_positive_rate(true_drift_points, detected_drift_points, tolerance)
    
    if p + r == 0:
        return 0.0
    
    return 2 * p * r / (p + r)


def evaluate_detector(
    true_drift_points: List[int],
    detected_drift_points: List[int],
    total_samples: int,
    tolerance: int = 50
) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        true_drift_points: Ground truth drift indices
        detected_drift_points: Detected drift indices
        total_samples: Total stream length
        tolerance: Detection tolerance window
        
    Returns:
        Dict with all metrics
    """
    avg_delay, delays = detection_delay(
        true_drift_points, detected_drift_points, tolerance
    )
    
    return {
        'detection_delay': avg_delay,
        'delays': delays,
        'false_positive_rate': false_positive_rate(
            true_drift_points, detected_drift_points, total_samples, tolerance
        ),
        'true_positive_rate': true_positive_rate(
            true_drift_points, detected_drift_points, tolerance
        ),
        'precision': precision(
            true_drift_points, detected_drift_points, tolerance
        ),
        'f1_score': f1_score(
            true_drift_points, detected_drift_points, tolerance
        ),
        'n_true_drifts': len(true_drift_points),
        'n_detected': len(detected_drift_points),
    }


class EnergyProfiler:
    """
    Profile energy consumption of different algorithms.
    
    Uses approximate energy costs per operation based on
    published data for ARM Cortex-M4 processors.
    """
    
    # Approximate energy per operation (picojoules)
    ENERGY_PER_OP = {
        'float_add': 3.7,
        'float_mul': 7.5,
        'float_div': 45.0,
        'int_add': 0.5,
        'int_mul': 2.0,
        'bitwise': 0.1,
        'popcount': 0.3,  # With hardware instruction
        'memory_access': 1.5,
    }
    
    def __init__(self):
        self.op_counts = {}
        self.total_energy_pj = 0.0
        
    def log_operation(self, op_type: str, count: int = 1):
        """Log operation counts."""
        self.op_counts[op_type] = self.op_counts.get(op_type, 0) + count
        self.total_energy_pj += self.ENERGY_PER_OP.get(op_type, 1.0) * count
    
    def get_summary(self) -> dict:
        """Get profiling summary."""
        return {
            'operation_counts': self.op_counts.copy(),
            'total_energy_pj': self.total_energy_pj,
            'total_energy_uj': self.total_energy_pj / 1e6,
            'total_energy_mj': self.total_energy_pj / 1e9,
        }
    
    @staticmethod
    def estimate_ths_energy(dim: int, n_edges: int) -> float:
        """
        Estimate energy for one THS inference step.
        
        Operations per edge:
        - 2 XOR operations (dim bits each): 2 * dim * 0.1 pJ
        - 1 popcount: dim * 0.3 pJ
        
        Args:
            dim: Hypervector dimension
            n_edges: Number of graph edges
            
        Returns:
            Energy in microjoules
        """
        per_edge = 2 * dim * 0.1 + dim * 0.3  # pJ
        total_pj = n_edges * per_edge
        return total_pj / 1e6  # Convert to uJ
    
    @staticmethod
    def estimate_autoencoder_energy(input_dim: int, hidden_dims: List[int]) -> float:
        """
        Estimate energy for one autoencoder inference.
        
        Operations: matrix multiplications through layers.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            
        Returns:
            Energy in microjoules
        """
        dims = [input_dim] + hidden_dims + [input_dim]
        total_macs = 0
        
        for i in range(len(dims) - 1):
            # MACs for this layer
            total_macs += dims[i] * dims[i+1]
        
        # Each MAC is roughly 1 add + 1 mul
        energy_pj = total_macs * (3.7 + 7.5)
        return energy_pj / 1e6  # Convert to uJ
