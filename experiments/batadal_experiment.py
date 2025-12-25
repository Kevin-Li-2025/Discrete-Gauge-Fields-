"""
BATADAL ICS Security Experiment
Water Distribution System Anomaly Detection

Dataset: BATADAL (BATtle of the Attack Detection ALgorithms)
- 43 sensors/actuators
- Train1: 8762 samples (normal operation)
- Train2: ~4400 samples (with labeled attacks)
- ATT_FLAG: 0=normal, 1=attack
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# ============================================================================
# DATA LOADING
# ============================================================================

def load_batadal(data_dir: str = "../data"):
    """Load BATADAL datasets."""
    train1_path = f"{data_dir}/BATADAL_train1.csv"
    train2_path = f"{data_dir}/BATADAL_train2.csv"
    
    # Load training data (normal operation)
    print("Loading BATADAL_train1.csv (normal operation)...")
    train1 = pd.read_csv(train1_path)
    train1.columns = [c.strip() for c in train1.columns]  # Strip whitespace
    
    # Load training data 2 (with attacks)
    print("Loading BATADAL_train2.csv (with attacks)...")
    try:
        train2 = pd.read_csv(train2_path)
        train2.columns = [c.strip() for c in train2.columns]  # Strip whitespace
    except:
        print("Train2 not found, using train1 with synthetic attacks")
        train2 = None
    
    # Extract sensor columns (exclude DATETIME and ATT_FLAG)
    sensor_cols = [c for c in train1.columns if c not in ['DATETIME', 'ATT_FLAG']]
    print(f"Sensors: {len(sensor_cols)}")
    print(f"Train1 samples: {len(train1)}")
    
    # Build graph: connect sensors in same physical group
    edges = build_water_network_graph(sensor_cols)
    print(f"Edges: {len(edges)}")
    
    # Convert to numpy (use train1 for training)
    data_train = train1[sensor_cols].values
    
    if train2 is not None and all(c in train2.columns for c in sensor_cols):
        data_test = train2[sensor_cols].values
        labels_test = train2['ATT_FLAG'].values if 'ATT_FLAG' in train2.columns else np.zeros(len(train2))
        # ATT_FLAG=-999 means unlabeled, treat as normal for now
        labels_test = np.where(labels_test == -999, 0, labels_test)
        labels_test = np.where(labels_test > 0, 1, 0)  # Binary
    else:
        # Split train1 and inject synthetic attacks
        data_test = data_train[7000:]
        data_train = data_train[:7000]
        labels_test = np.zeros(len(data_test))
        # Inject attack in second half
        attack_start = len(data_test) // 2
        data_test, labels_test = inject_synthetic_attack(data_test, labels_test, attack_start)
    
    return data_train, data_test, labels_test, edges, sensor_cols


def build_water_network_graph(sensor_cols: List[str]) -> List[Tuple[int, int]]:
    """
    Build graph based on water network topology.
    Connect sensors in logical groups (tanks, pumps, valves, pressures).
    """
    edges = []
    n = len(sensor_cols)
    
    # Group sensors by prefix
    groups = {}
    for i, col in enumerate(sensor_cols):
        prefix = col.split('_')[0]  # L, F, S, P
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(i)
    
    # Connect sensors within same group (chain)
    for prefix, indices in groups.items():
        for i in range(len(indices) - 1):
            edges.append((indices[i], indices[i+1]))
    
    # Connect related sensors across groups
    # Tank levels (L) connect to their pump flows (F)
    for i, col in enumerate(sensor_cols):
        if col.startswith('L_T'):
            tank_num = col.split('T')[1]
            # Find corresponding pump
            for j, col2 in enumerate(sensor_cols):
                if col2.startswith('F_PU') and i != j:
                    if (i, j) not in edges and (j, i) not in edges:
                        edges.append((i, j))
                    break
    
    # Ensure connectivity
    if len(edges) < n - 1:
        # Add chain to ensure connected
        for i in range(n - 1):
            if (i, i+1) not in edges and (i+1, i) not in edges:
                edges.append((i, i+1))
    
    return edges[:60]  # Limit edges for efficiency


def inject_synthetic_attack(data: np.ndarray, labels: np.ndarray, start: int) -> Tuple[np.ndarray, np.ndarray]:
    """Inject synthetic attack: manipulate tank levels and pump statuses."""
    data = data.copy()
    labels = labels.copy()
    
    # Attack: gradually increase tank levels (stealthy)
    n_sensors = data.shape[1]
    for t in range(start, len(data)):
        progress = (t - start) / (len(data) - start)
        # Manipulate first 3 sensors (tank levels)
        for s in range(min(3, n_sensors)):
            data[t, s] += progress * 0.5 * np.std(data[:start, s])
        labels[t] = 1
    
    return data, labels


# ============================================================================
# THS DETECTOR
# ============================================================================

class THS:
    """THS detector for BATADAL."""
    
    def __init__(self, dim: int = 1000, seed: int = 42):
        self.dim = dim
        self.seed = seed
    
    def fit(self, data: np.ndarray, edges: List[Tuple[int, int]]):
        np.random.seed(self.seed)
        
        self.edges = edges
        self.n_sensors = data.shape[1]
        self.P = np.random.randn(self.dim, self.n_sensors)
        
        # Z-score normalize
        self.train_mean = np.mean(data, axis=0)
        self.train_std = np.std(data, axis=0) + 1e-6
        
        enc = self._encode(data)
        
        # Learn reference
        self.ref = {}
        for u, v in edges:
            xors = np.bitwise_xor(enc[:, u, :], enc[:, v, :])
            self.ref[(u, v)] = (np.sum(xors, axis=0) > len(data)/2).astype(np.uint8)
        
        # Baseline
        energies = self._energy(enc)
        self.mu = np.mean(energies)
        self.sigma = np.std(energies) + 1e-6
        
        return self
    
    def _encode(self, data: np.ndarray) -> np.ndarray:
        d_norm = (data - self.train_mean) / self.train_std
        # Project each sensor's value
        enc = np.zeros((len(data), self.n_sensors, self.dim), dtype=np.uint8)
        for t in range(len(data)):
            for s in range(self.n_sensors):
                enc[t, s] = (d_norm[t, s] * self.P[:, s] > 0).astype(np.uint8)
        return enc
    
    def _energy(self, enc: np.ndarray) -> np.ndarray:
        total = np.zeros(len(enc))
        for (u, v), C in self.ref.items():
            total += np.sum(np.bitwise_xor(np.bitwise_xor(enc[:,u,:], enc[:,v,:]), C), axis=1)
        return total / (len(self.ref) * self.dim)
    
    def score(self, data: np.ndarray) -> np.ndarray:
        enc = self._encode(data)
        return (self._energy(enc) - self.mu) / self.sigma


class CUSUM:
    """CUSUM baseline."""
    
    def fit(self, data, edges=None):
        self.m = np.mean(data, axis=0)
        self.s = np.std(data, axis=0) + 1e-6
        return self
    
    def score(self, data):
        z = (data - self.m) / self.s
        S = np.zeros(data.shape[1])
        scores = []
        for t in range(len(data)):
            S = np.maximum(0, S + np.abs(z[t]) - 0.5)
            scores.append(np.max(S))
        return np.array(scores)


class Laplacian:
    """Laplacian baseline (THS with C_e = 0)."""
    
    def __init__(self, dim: int = 1000, seed: int = 42):
        self.dim = dim
        self.seed = seed
    
    def fit(self, data, edges):
        np.random.seed(self.seed)
        self.edges = edges
        self.n_sensors = data.shape[1]
        self.P = np.random.randn(self.dim, self.n_sensors)
        self.train_mean = np.mean(data, axis=0)
        self.train_std = np.std(data, axis=0) + 1e-6
        
        enc = self._encode(data)
        energies = self._energy(enc)
        self.mu = np.mean(energies)
        self.sigma = np.std(energies) + 1e-6
        return self
    
    def _encode(self, data):
        d_norm = (data - self.train_mean) / self.train_std
        enc = np.zeros((len(data), self.n_sensors, self.dim), dtype=np.uint8)
        for t in range(len(data)):
            for s in range(self.n_sensors):
                enc[t, s] = (d_norm[t, s] * self.P[:, s] > 0).astype(np.uint8)
        return enc
    
    def _energy(self, enc):
        total = np.zeros(len(enc))
        for u, v in self.edges:
            total += np.sum(np.bitwise_xor(enc[:,u,:], enc[:,v,:]), axis=1)
        return total / (len(self.edges) * self.dim)
    
    def score(self, data):
        enc = self._encode(data)
        return (self._energy(enc) - self.mu) / self.sigma


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_detection(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Evaluate detection performance."""
    predictions = (scores > threshold).astype(int)
    
    # Metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    tpr = tp / (tp + fn + 1e-6)
    fpr = fp / (fp + tn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    f1 = 2 * precision * tpr / (precision + tpr + 1e-6)
    
    # Detection delay
    attack_starts = np.where(np.diff(labels.astype(int)) == 1)[0] + 1
    detection_times = np.where(predictions == 1)[0]
    
    delays = []
    for attack_start in attack_starts:
        detections_after = detection_times[detection_times >= attack_start]
        if len(detections_after) > 0:
            delays.append(detections_after[0] - attack_start)
    
    avg_delay = np.mean(delays) if delays else float('inf')
    
    return {
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'f1': f1,
        'delay': avg_delay
    }


def run_experiment():
    """Run BATADAL experiment."""
    print("=" * 60)
    print("BATADAL ICS Security Experiment")
    print("=" * 60)
    
    # Load data
    train_data, test_data, test_labels, edges, sensor_cols = load_batadal()
    
    print(f"\nTrain: {train_data.shape}")
    print(f"Test: {test_data.shape}")
    print(f"Attack samples: {np.sum(test_labels)}")
    
    # Detectors
    detectors = {
        'THS': THS(dim=1000),
        'CUSUM': CUSUM(),
        'Laplacian': Laplacian(dim=1000),
    }
    
    results = {}
    
    for name, det in detectors.items():
        print(f"\n{name}:")
        det.fit(train_data, edges)
        
        # Score test data
        test_scores = det.score(test_data)
        train_scores = det.score(train_data)
        
        # Set threshold at 95th percentile of training
        threshold = np.percentile(train_scores, 95)
        
        # Evaluate
        metrics = evaluate_detection(test_scores, test_labels, threshold)
        results[name] = metrics
        
        print(f"  TPR: {metrics['tpr']:.3f}")
        print(f"  FPR: {metrics['fpr']:.3f}")
        print(f"  F1:  {metrics['f1']:.3f}")
        print(f"  Delay: {metrics['delay']:.1f}")
    
    # LaTeX table
    print("\n" + "=" * 60)
    print("LATEX TABLE")
    print("=" * 60)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{BATADAL ICS Attack Detection (43 sensors)}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Method & TPR & FPR & F1 \\")
    print(r"\midrule")
    for name in detectors:
        r = results[name]
        print(f"{name} & {r['tpr']:.2f} & {r['fpr']:.2f} & {r['f1']:.2f} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
