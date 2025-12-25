"""
TPAMI Final Experiment: THS vs All Baselines
Results for paper Section 4: Experiments
"""
import numpy as np
import sys
sys.path.insert(0, '../baselines')

np.random.seed(42)

# ============================================================================
# THS (Fixed encoding with z-score normalization)
# ============================================================================

class THS:
    """Topological Hypervector Sheaves detector."""
    
    def __init__(self, dim=1000):
        self.dim = dim
    
    def fit(self, data, edges):
        np.random.seed(42)
        self.edges = edges
        self.n_sensors = data.shape[1]
        self.P = np.random.randn(self.dim)
        
        # Z-score normalize using training stats
        self.train_mean = np.mean(data, axis=0)
        self.train_std = np.std(data, axis=0) + 1e-6
        
        enc = self._encode(data)
        
        # Learn reference connection
        self.ref = {}
        for u, v in edges:
            xors = np.bitwise_xor(enc[:, u, :], enc[:, v, :])
            self.ref[(u, v)] = (np.sum(xors, axis=0) > len(data)/2).astype(np.uint8)
        
        # Baseline statistics
        energies = self._energy(enc)
        self.mu, self.sigma = np.mean(energies), np.std(energies) + 1e-6
        return self
    
    def _encode(self, data):
        d_norm = (data - self.train_mean) / self.train_std
        enc = np.zeros((len(data), self.n_sensors, self.dim), dtype=np.uint8)
        for t in range(len(data)):
            for s in range(self.n_sensors):
                enc[t, s] = (d_norm[t, s] * self.P > 0).astype(np.uint8)
        return enc
    
    def _energy(self, enc):
        total = np.zeros(len(enc))
        for (u,v), C in self.ref.items():
            total += np.sum(np.bitwise_xor(np.bitwise_xor(enc[:,u,:], enc[:,v,:]), C), axis=1)
        return total / (len(self.ref) * self.dim)
    
    def score(self, data):
        enc = self._encode(data)
        return (self._energy(enc) - self.mu) / self.sigma


class Laplacian:
    """Graph Laplacian baseline (THS with C_e = 0)."""
    
    def __init__(self, dim=1000):
        self.dim = dim
    
    def fit(self, data, edges):
        np.random.seed(42)
        self.edges = edges
        self.n_sensors = data.shape[1]
        self.P = np.random.randn(self.dim)
        self.train_mean = np.mean(data, axis=0)
        self.train_std = np.std(data, axis=0) + 1e-6
        
        enc = self._encode(data)
        energies = self._energy(enc)
        self.mu, self.sigma = np.mean(energies), np.std(energies) + 1e-6
        return self
    
    def _encode(self, data):
        d_norm = (data - self.train_mean) / self.train_std
        enc = np.zeros((len(data), self.n_sensors, self.dim), dtype=np.uint8)
        for t in range(len(data)):
            for s in range(self.n_sensors):
                enc[t, s] = (d_norm[t, s] * self.P > 0).astype(np.uint8)
        return enc
    
    def _energy(self, enc):
        total = np.zeros(len(enc))
        for u,v in self.edges:
            total += np.sum(np.bitwise_xor(enc[:,u,:], enc[:,v,:]), axis=1)
        return total / (len(self.edges) * self.dim)
    
    def score(self, data):
        enc = self._encode(data)
        return (self._energy(enc) - self.mu) / self.sigma


class CUSUM:
    """Mean-shift CUSUM baseline."""
    
    def fit(self, data, edges=None):
        self.m = np.mean(data, 0)
        self.s = np.std(data, 0) + 1e-6
        return self
    
    def score(self, data):
        z = (data - self.m) / self.s
        S = np.zeros(data.shape[1])
        scores = []
        for t in range(len(data)):
            S = np.maximum(0, S + np.abs(z[t]) - 0.5)
            scores.append(np.max(S))
        return np.array(scores)


class ADWIN:
    """ADWIN-style per-sensor z-score."""
    
    def fit(self, data, edges=None):
        self.m = np.mean(data, 0)
        self.s = np.std(data, 0) + 1e-6
        return self
    
    def score(self, data):
        z = (data - self.m) / self.s
        return np.max(np.abs(z), axis=1)


# ============================================================================
# GCN-AE (if PyTorch available)
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class GCNAE(nn.Module):
        def __init__(self, n_nodes, hidden_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 16))
            self.decoder = nn.Sequential(nn.Linear(16, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
        def forward(self, x, adj):
            if x.dim() == 2: x = x.unsqueeze(-1)
            x_agg = torch.matmul(adj, x)
            z = self.encoder(x_agg)
            return self.decoder(z).squeeze(-1)
    
    class GCNAEDetector:
        def __init__(self, epochs=50, hidden_dim=32):
            self.epochs = epochs
            self.hidden_dim = hidden_dim
        
        def fit(self, data, edges):
            n = data.shape[1]
            adj = np.zeros((n, n))
            for i, j in edges: adj[i, j] = adj[j, i] = 1
            adj = adj + np.eye(n)
            D = np.diag(1.0 / np.sqrt(np.sum(adj, axis=1)))
            adj = D @ adj @ D
            self.adj = torch.FloatTensor(adj)
            
            self.model = GCNAE(n, self.hidden_dim)
            opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
            X = torch.FloatTensor(data)
            
            self.model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = F.mse_loss(self.model(X, self.adj), X)
                loss.backward()
                opt.step()
            
            self.model.eval()
            with torch.no_grad():
                errors = torch.mean((self.model(X, self.adj) - X)**2, dim=1).numpy()
            self.mu, self.sigma = np.mean(errors), np.std(errors) + 1e-6
            return self
        
        def score(self, data):
            X = torch.FloatTensor(data)
            self.model.eval()
            with torch.no_grad():
                errors = torch.mean((self.model(X, self.adj) - X)**2, dim=1).numpy()
            return (errors - self.mu) / self.sigma
    
    GCN_AVAILABLE = True
except:
    GCN_AVAILABLE = False


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_data(n_sensors=20, n_samples=1000, seed=42):
    np.random.seed(seed)
    pos = np.random.rand(n_sensors, 2) * 100
    edges = [(i, j) for i in range(n_sensors) for j in range(i+1, n_sensors) 
             if np.linalg.norm(pos[i] - pos[j]) < 40]
    
    data = np.zeros((n_samples, n_sensors))
    for s in range(n_sensors):
        phase = s * 0.4
        amp = 20 + s * 2
        data[:, s] = 50 + amp * np.sin(2*np.pi*np.arange(n_samples)/100 + phase) + np.random.randn(n_samples)*3
    
    return data, edges


def shuffle_drift(d, s):
    """Relational drift: shuffle sensor assignments."""
    d = d.copy()
    perm = np.random.permutation(d.shape[1])
    d[s:] = d[s:, perm]
    return d


def decorrelate_drift(d, s, n_affected=5):
    """Relational drift: randomize subset of sensors."""
    d = d.copy()
    for i in range(n_affected):
        post = d[s:, i].copy()
        np.random.shuffle(post)
        d[s:, i] = post
    return d


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment():
    print("=" * 60)
    print("TPAMI Final Experiment: THS vs All Baselines")
    print("=" * 60)
    
    data, edges = generate_data(n_sensors=20, n_samples=1000)
    print(f"Data: {data.shape[1]} sensors, {data.shape[0]} samples, {len(edges)} edges")
    
    train = data[:700]
    test = data[700:]
    drift_start = 150
    n_seeds = 20
    
    detectors = {
        'THS': lambda: THS(dim=1000),
        'Laplacian': lambda: Laplacian(dim=1000),
        'CUSUM': lambda: CUSUM(),
        'ADWIN': lambda: ADWIN(),
    }
    
    if GCN_AVAILABLE:
        detectors['GCN-AE'] = lambda: GCNAEDetector(epochs=30)
    
    drift_types = {
        'Shuffle': shuffle_drift,
        'Decorrelate': lambda d, s: decorrelate_drift(d, s, 5),
    }
    
    results = {}
    
    for drift_name, drift_fn in drift_types.items():
        print(f"\n--- {drift_name} Drift ---")
        results[drift_name] = {}
        
        for det_name, det_fn in detectors.items():
            tprs, delays = [], []
            
            for seed in range(n_seeds):
                np.random.seed(seed + 100)
                td = drift_fn(test, drift_start)
                
                det = det_fn()
                det.fit(train, edges)
                
                train_scores = det.score(train)
                thresh = np.percentile(train_scores, 95)
                
                test_scores = det.score(td)
                
                # Detection with persistence (3 consecutive)
                detected, det_time = False, None
                consec = 0
                for t, s in enumerate(test_scores):
                    if s > thresh:
                        consec += 1
                        if consec >= 3 and t >= drift_start and det_time is None:
                            detected = True
                            det_time = t
                    else:
                        consec = 0
                
                tprs.append(1.0 if detected else 0.0)
                delays.append(det_time - drift_start if det_time else 999)
            
            results[drift_name][det_name] = {
                'tpr': np.mean(tprs),
                'delay': np.mean([d for d in delays if d < 999]) if any(d < 999 for d in delays) else float('inf')
            }
            print(f"  {det_name:12s} TPR={np.mean(tprs):.2f} Delay={results[drift_name][det_name]['delay']:.0f}")
    
    # Generate LaTeX
    print("\n" + "=" * 60)
    print("LATEX TABLE")
    print("=" * 60)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Comparison with All Baselines}")
    print(r"\begin{tabular}{l" + "c" * len(detectors) + "}")
    print(r"\toprule")
    print("Drift Type & " + " & ".join(detectors.keys()) + r" \\")
    print(r"\midrule")
    for drift_name in drift_types:
        row = drift_name
        for det_name in detectors:
            row += f" & {results[drift_name][det_name]['tpr']:.2f}"
        print(row + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
