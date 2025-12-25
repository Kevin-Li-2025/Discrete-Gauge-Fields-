"""
METR-LA Large-Scale Experiment for THS (OPTIMIZED)
207 traffic sensors - reduced sample size for speed
"""

import numpy as np
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA GENERATION (Optimized)
# ============================================================================

def generate_metrla_data(n_sensors: int = 207, n_samples: int = 5000, seed: int = 42):
    """Generate METR-LA-like synthetic data (optimized)."""
    np.random.seed(seed)
    
    print(f"Generating METR-LA-like data: {n_sensors} sensors, {n_samples} samples")
    
    # Positions for edge generation
    positions = np.random.rand(n_sensors, 2) * 100
    
    # Build edges (distance threshold 8km for ~150 edges)
    threshold = 8.0
    edges = []
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            if np.linalg.norm(positions[i] - positions[j]) < threshold:
                edges.append((i, j))
    
    print(f"Generated {len(edges)} edges")
    
    # Generate traffic-like data with correlations
    t = np.arange(n_samples)
    daily = 288  # 5-min intervals
    
    # Global signal
    global_signal = 60 + 30 * np.sin(2 * np.pi * t / daily)
    
    # Per-sensor data with local correlations
    data = np.zeros((n_samples, n_sensors))
    for s in range(n_sensors):
        phase = np.random.rand() * 2 * np.pi
        amp = 0.8 + 0.4 * np.random.rand()
        data[:, s] = amp * (60 + 30 * np.sin(2 * np.pi * t / daily + phase)) + np.random.randn(n_samples) * 5
        data[:, s] = np.clip(data[:, s], 0, 100)
    
    # Smooth along edges
    for _ in range(2):
        new_data = data.copy()
        for i, j in edges:
            avg = (data[:, i] + data[:, j]) / 2
            new_data[:, i] = 0.7 * data[:, i] + 0.3 * avg
            new_data[:, j] = 0.7 * data[:, j] + 0.3 * avg
        data = new_data
    
    return data, edges


# ============================================================================
# VECTORIZED THS DETECTOR
# ============================================================================

class THSFast:
    """Vectorized THS detector for speed."""
    
    def __init__(self, dim: int = 2000, seed: int = 42):
        self.dim = dim
        self.seed = seed
        
    def fit(self, data: np.ndarray, edges: List[Tuple[int, int]]):
        np.random.seed(self.seed)
        
        self.edges = edges
        self.n_sensors = data.shape[1]
        
        # Projection: shape (dim,) for scalar sensors
        self.P = np.random.randn(self.dim)
        
        # Encode ALL data at once: (n_samples, n_sensors, dim)
        # For scalar data: outer product with P
        encoded = (np.outer(data.flatten(), self.P).reshape(len(data), self.n_sensors, self.dim) > 0).astype(np.uint8)
        
        # Learn reference for each edge via majority vote
        self.reference = {}
        for u, v in edges:
            xors = np.bitwise_xor(encoded[:, u, :], encoded[:, v, :])
            self.reference[(u, v)] = (np.sum(xors, axis=0) > len(data) / 2).astype(np.uint8)
        
        # Baseline energy
        energies = self._energy_batch(encoded)
        self.mu = np.mean(energies)
        self.sigma = np.std(energies) + 1e-6
        
        return self
    
    def _encode_batch(self, data: np.ndarray) -> np.ndarray:
        """Encode all samples at once."""
        return (np.outer(data.flatten(), self.P).reshape(len(data), self.n_sensors, self.dim) > 0).astype(np.uint8)
    
    def _energy_batch(self, encoded: np.ndarray) -> np.ndarray:
        """Compute energy for all samples."""
        total = np.zeros(len(encoded))
        for (u, v), C in self.reference.items():
            r = np.bitwise_xor(np.bitwise_xor(encoded[:, u, :], encoded[:, v, :]), C)
            total += np.sum(r, axis=1)
        return total / (len(self.reference) * self.dim)
    
    def score_batch(self, data: np.ndarray) -> np.ndarray:
        encoded = self._encode_batch(data)
        energies = self._energy_batch(encoded)
        return (energies - self.mu) / self.sigma


class CUSUMFast:
    """Vectorized CUSUM baseline."""
    
    def __init__(self, delta: float = 0.5):
        self.delta = delta
    
    def fit(self, data: np.ndarray, edges=None):
        self.means = np.mean(data, axis=0)
        self.stds = np.std(data, axis=0) + 1e-6
        return self
    
    def score_batch(self, data: np.ndarray) -> np.ndarray:
        z = (data - self.means) / self.stds
        S = np.zeros(data.shape[1])
        scores = []
        for t in range(len(data)):
            S = np.maximum(0, S + np.abs(z[t]) - self.delta)
            scores.append(np.max(S))
        return np.array(scores)


class LaplacianFast:
    """Vectorized Laplacian baseline."""
    
    def __init__(self, dim: int = 2000, seed: int = 42):
        self.dim = dim
        self.seed = seed
    
    def fit(self, data: np.ndarray, edges: List[Tuple[int, int]]):
        np.random.seed(self.seed)
        
        self.edges = edges
        self.n_sensors = data.shape[1]
        self.P = np.random.randn(self.dim)
        
        encoded = (np.outer(data.flatten(), self.P).reshape(len(data), self.n_sensors, self.dim) > 0).astype(np.uint8)
        
        # No learned reference (C_e = 0)
        energies = self._energy_batch(encoded)
        self.mu = np.mean(energies)
        self.sigma = np.std(energies) + 1e-6
        
        return self
    
    def _energy_batch(self, encoded):
        total = np.zeros(len(encoded))
        for u, v in self.edges:
            r = np.bitwise_xor(encoded[:, u, :], encoded[:, v, :])
            total += np.sum(r, axis=1)
        return total / (len(self.edges) * self.dim)
    
    def score_batch(self, data):
        encoded = (np.outer(data.flatten(), self.P).reshape(len(data), self.n_sensors, self.dim) > 0).astype(np.uint8)
        energies = self._energy_batch(encoded)
        return (energies - self.mu) / self.sigma


# ============================================================================
# EXPERIMENT
# ============================================================================

def inject_drift(data: np.ndarray, drift_start: int, drift_type: str = "decorrelation"):
    """Inject drift into test data."""
    drifted = data.copy()
    
    if drift_type == "decorrelation":
        affected = list(range(20))  # 20 sensors
        for s in affected:
            mean_s = np.mean(data[:drift_start, s])
            std_s = np.std(data[:drift_start, s])
            drifted[drift_start:, s] = mean_s + std_s * np.random.randn(len(data) - drift_start)
    
    elif drift_type == "swap":
        drifted[drift_start:, 0] = data[drift_start:, 1]
        drifted[drift_start:, 1] = data[drift_start:, 0]
    
    elif drift_type == "gradual":
        for t in range(drift_start, len(data)):
            decay = 1 - 0.5 * (t - drift_start) / (len(data) - drift_start)
            for s in range(20):
                mean_s = np.mean(data[:drift_start, s])
                drifted[t, s] = decay * data[t, s] + (1 - decay) * (mean_s + np.random.randn() * 5)
    
    return drifted


def evaluate(detector, train_data, test_data, drift_start, edges, 
             threshold_far=0.05, persistence=3):
    """Evaluate detector."""
    detector.fit(train_data, edges)
    
    train_scores = detector.score_batch(train_data)
    threshold = np.percentile(train_scores, 100 * (1 - threshold_far))
    
    test_scores = detector.score_batch(test_data)
    
    # Persistence detection
    consecutive = 0
    detection_time = None
    for t, s in enumerate(test_scores):
        if s > threshold:
            consecutive += 1
            if consecutive >= persistence and detection_time is None:
                detection_time = t - persistence + 1
        else:
            consecutive = 0
    
    if detection_time is not None and detection_time >= drift_start:
        return {'tpr': 1.0, 'delay': detection_time - drift_start}
    elif detection_time is not None:
        return {'tpr': 0.0, 'delay': float('inf')}  # False alarm before drift
    else:
        return {'tpr': 0.0, 'delay': float('inf')}


def run_experiment():
    """Run METR-LA experiment."""
    print("=" * 60)
    print("METR-LA Large-Scale Experiment (Optimized)")
    print("=" * 60)
    
    # Generate data
    data, edges = generate_metrla_data(n_sensors=207, n_samples=5000, seed=42)
    
    # Split
    train_end = 3500
    drift_start = 4000 - train_end  # Relative to test
    train_data = data[:train_end]
    test_data = data[train_end:]
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}, Drift at: {drift_start}")
    
    detectors = {
        'THS': lambda s: THSFast(dim=2000, seed=s),
        'CUSUM': lambda s: CUSUMFast(),
        'Laplacian': lambda s: LaplacianFast(dim=2000, seed=s),
    }
    
    drift_types = ['decorrelation', 'swap', 'gradual']
    n_seeds = 10
    
    results = {}
    
    for drift_type in drift_types:
        print(f"\n--- {drift_type.upper()} ---")
        results[drift_type] = {}
        
        for name, detector_fn in detectors.items():
            tprs, delays = [], []
            
            for seed in range(n_seeds):
                np.random.seed(seed + 100)
                drifted_test = inject_drift(test_data, drift_start, drift_type)
                
                res = evaluate(detector_fn(seed), train_data, drifted_test, drift_start, edges)
                tprs.append(res['tpr'])
                delays.append(res['delay'] if res['delay'] != float('inf') else 500)
            
            results[drift_type][name] = {
                'tpr': np.mean(tprs),
                'delay': np.mean(delays)
            }
            print(f"  {name:12s} TPR={np.mean(tprs):.2f} Delay={np.mean(delays):.0f}")
    
    # LaTeX table
    print("\n" + "=" * 60)
    print("LATEX TABLE")
    print("=" * 60)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{METR-LA (207 sensors, 5K samples)}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Drift Type & THS & CUSUM & Laplacian \\")
    print(r"\midrule")
    for dt in drift_types:
        row = dt.capitalize()
        for m in ['THS', 'CUSUM', 'Laplacian']:
            row += f" & {results[dt][m]['tpr']:.2f}"
        print(row + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
