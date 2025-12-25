"""
Full Comparison: THS vs DL Baselines
"""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../baselines')

import numpy as np
import torch
torch.manual_seed(42)

# Try import DL baselines
try:
    from dl_baselines import GCNAEDetector, TransformerDetector, SheafDetector
    DL_AVAILABLE = True
except:
    DL_AVAILABLE = False
    print("DL baselines not available")

# ============================================================================
# THS and Simple Baselines (from metrla_quick.py)
# ============================================================================

class THSQuick:
    def __init__(self, dim=1000):
        self.dim = dim
    
    def fit(self, data, edges):
        np.random.seed(42)
        self.edges = edges
        self.n_sensors = data.shape[1]
        self.P = np.random.randn(self.dim)
        
        enc = (np.outer(data.flatten(), self.P).reshape(len(data), self.n_sensors, self.dim) > 0).astype(np.uint8)
        self.ref = {(u,v): (np.sum(np.bitwise_xor(enc[:,u,:], enc[:,v,:]), axis=0) > len(data)/2).astype(np.uint8) 
                    for u,v in edges}
        
        energies = self._energy(enc)
        self.mu, self.sigma = np.mean(energies), np.std(energies) + 1e-6
        return self
    
    def _energy(self, enc):
        total = np.zeros(len(enc))
        for (u,v), C in self.ref.items():
            total += np.sum(np.bitwise_xor(np.bitwise_xor(enc[:,u,:], enc[:,v,:]), C), axis=1)
        return total / (len(self.ref) * self.dim)
    
    def score_batch(self, data):
        enc = (np.outer(data.flatten(), self.P).reshape(len(data), self.n_sensors, self.dim) > 0).astype(np.uint8)
        return (self._energy(enc) - self.mu) / self.sigma

class CUSUMQuick:
    def fit(self, data, edges=None):
        self.means = np.mean(data, axis=0)
        self.stds = np.std(data, axis=0) + 1e-6
        return self
    
    def score_batch(self, data):
        z = (data - self.means) / self.stds
        S = np.zeros(data.shape[1])
        scores = []
        for t in range(len(data)):
            S = np.maximum(0, S + np.abs(z[t]) - 0.5)
            scores.append(np.max(S))
        return np.array(scores)

class LaplacianQuick:
    def __init__(self, dim=1000):
        self.dim = dim
    
    def fit(self, data, edges):
        np.random.seed(42)
        self.edges = edges
        self.n_sensors = data.shape[1]
        self.P = np.random.randn(self.dim)
        
        enc = (np.outer(data.flatten(), self.P).reshape(len(data), self.n_sensors, self.dim) > 0).astype(np.uint8)
        energies = self._energy(enc)
        self.mu, self.sigma = np.mean(energies), np.std(energies) + 1e-6
        return self
    
    def _energy(self, enc):
        total = np.zeros(len(enc))
        for u,v in self.edges:
            total += np.sum(np.bitwise_xor(enc[:,u,:], enc[:,v,:]), axis=1)
        return total / (len(self.edges) * self.dim)
    
    def score_batch(self, data):
        enc = (np.outer(data.flatten(), self.P).reshape(len(data), self.n_sensors, self.dim) > 0).astype(np.uint8)
        return (self._energy(enc) - self.mu) / self.sigma

# ============================================================================
# DATA
# ============================================================================

def generate_data(n_sensors=30, n_samples=1000, seed=42):
    np.random.seed(seed)
    pos = np.random.rand(n_sensors, 2) * 100
    edges = [(i,j) for i in range(n_sensors) for j in range(i+1, n_sensors) 
             if np.linalg.norm(pos[i]-pos[j]) < 30]
    
    t = np.arange(n_samples)
    data = np.array([60 + 30*np.sin(2*np.pi*t/288 + np.random.rand()*2*np.pi) + np.random.randn(n_samples)*5 
                     for _ in range(n_sensors)]).T
    
    # Add edge correlations
    for _ in range(2):
        new_data = data.copy()
        for i, j in edges:
            avg = (data[:, i] + data[:, j]) / 2
            new_data[:, i] = 0.7 * data[:, i] + 0.3 * avg
            new_data[:, j] = 0.7 * data[:, j] + 0.3 * avg
        data = new_data
    
    return data, edges

def inject_drift(data, drift_start):
    d = data.copy()
    for s in range(5):  # Decorrelate 5 sensors
        d[drift_start:, s] = np.mean(data[:drift_start, s]) + np.std(data[:drift_start, s]) * np.random.randn(len(data)-drift_start)
    return d

# ============================================================================
# EXPERIMENT
# ============================================================================

def run():
    print("Full Comparison: THS vs All Baselines")
    print("=" * 60)
    
    data, edges = generate_data(n_sensors=30, n_samples=1000)
    print(f"Data: {data.shape[1]} sensors, {data.shape[0]} samples, {len(edges)} edges")
    
    train = data[:700]
    test = data[700:]
    drift_start = 150
    
    # Detectors
    detectors = {
        'THS': THSQuick(),
        'CUSUM': CUSUMQuick(),
        'Laplacian': LaplacianQuick(),
    }
    
    if DL_AVAILABLE:
        detectors['GCN-AE'] = GCNAEDetector(epochs=30)
        detectors['Sheaf'] = SheafDetector(epochs=30)
    
    n_seeds = 5
    results = {}
    
    for name, det in detectors.items():
        print(f"\nTesting {name}...")
        tprs = []
        
        for seed in range(n_seeds):
            np.random.seed(seed + 100)
            drifted = inject_drift(test, drift_start)
            
            det.fit(train, edges)
            scores = det.score_batch(drifted)
            
            train_scores = det.score_batch(train)
            thresh = np.percentile(train_scores, 95)
            
            detected = np.any(scores[drift_start:] > thresh)
            tprs.append(1.0 if detected else 0.0)
        
        results[name] = np.mean(tprs)
        print(f"  {name}: TPR = {np.mean(tprs):.2f}")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, tpr in results.items():
        print(f"{name:15s}: {tpr:.2f}")
    
    print("\n" + "=" * 60)
    print("LATEX TABLE")
    print("=" * 60)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Comparison with DL Baselines}")
    print(r"\begin{tabular}{lc}")
    print(r"\toprule")
    print(r"Method & TPR \\")
    print(r"\midrule")
    for name, tpr in results.items():
        print(f"{name} & {tpr:.2f} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    return results

if __name__ == "__main__":
    run()
