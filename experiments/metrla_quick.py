"""
METR-LA Quick Experiment - Minimal version for fast results
"""
import numpy as np
from typing import List, Tuple

def generate_data(n_sensors=50, n_samples=2000, seed=42):
    np.random.seed(seed)
    pos = np.random.rand(n_sensors, 2) * 100
    edges = [(i,j) for i in range(n_sensors) for j in range(i+1, n_sensors) 
             if np.linalg.norm(pos[i]-pos[j]) < 20]
    
    t = np.arange(n_samples)
    data = np.array([60 + 30*np.sin(2*np.pi*t/288 + np.random.rand()*2*np.pi) + np.random.randn(n_samples)*5 
                     for _ in range(n_sensors)]).T
    return data, edges

class THSQuick:
    def __init__(self, dim=1000):
        self.dim = dim
    
    def fit(self, data, edges):
        np.random.seed(42)
        self.edges = edges
        self.n_sensors = data.shape[1]
        self.P = np.random.randn(self.dim)
        
        # Encode
        enc = (np.outer(data.flatten(), self.P).reshape(len(data), self.n_sensors, self.dim) > 0).astype(np.uint8)
        
        # Learn reference
        self.ref = {(u,v): (np.sum(np.bitwise_xor(enc[:,u,:], enc[:,v,:]), axis=0) > len(data)/2).astype(np.uint8) 
                    for u,v in edges}
        
        # Baseline
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

def inject_drift(data, drift_start):
    d = data.copy()
    for s in range(10):  # Decorrelate 10 sensors
        d[drift_start:, s] = np.mean(data[:drift_start, s]) + np.std(data[:drift_start, s]) * np.random.randn(len(data)-drift_start)
    return d

def run():
    print("METR-LA Quick Experiment")
    print("=" * 50)
    
    data, edges = generate_data(n_sensors=50, n_samples=2000)
    print(f"Data: {data.shape[1]} sensors, {data.shape[0]} samples, {len(edges)} edges")
    
    train = data[:1400]
    test = data[1400:]
    drift_start = 300
    
    results = {}
    n_seeds = 10
    
    for name, det_class in [('THS', THSQuick), ('CUSUM', CUSUMQuick), ('Laplacian', LaplacianQuick)]:
        tprs = []
        for seed in range(n_seeds):
            np.random.seed(seed + 100)
            drifted = inject_drift(test, drift_start)
            
            det = det_class() if name != 'CUSUM' else det_class()
            det.fit(train, edges)
            scores = det.score_batch(drifted)
            
            # Threshold at 95th percentile of train
            train_scores = det.score_batch(train)
            thresh = np.percentile(train_scores, 95)
            
            # Check if detected after drift
            detected = np.any(scores[drift_start:] > thresh)
            tprs.append(1.0 if detected else 0.0)
        
        results[name] = np.mean(tprs)
        print(f"{name:12s} TPR = {np.mean(tprs):.2f}")
    
    print("\n" + "=" * 50)
    print("LATEX:")
    print(f"THS: {results['THS']:.2f}, CUSUM: {results['CUSUM']:.2f}, Laplacian: {results['Laplacian']:.2f}")
    
    return results

if __name__ == "__main__":
    run()
