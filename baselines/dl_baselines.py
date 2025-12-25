"""
Deep Learning Baselines for Drift Detection
- GCN Autoencoder
- Transformer
- Neural Sheaf (simplified)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Check for PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using simple alternatives.")


# ============================================================================
# GCN AUTOENCODER
# ============================================================================

class GCNLayer(nn.Module):
    """Simple Graph Convolutional Layer."""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, adj):
        # x: (batch, n_nodes, in_dim)
        # adj: (n_nodes, n_nodes) normalized adjacency
        out = torch.matmul(adj, x)  # aggregate neighbors
        out = self.linear(out)
        return F.relu(out)


class GCNAE(nn.Module):
    """GCN Autoencoder for anomaly detection."""
    
    def __init__(self, n_nodes, in_dim=1, hidden_dim=32, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.n_nodes = n_nodes
    
    def forward(self, x, adj):
        # x: (batch, n_nodes) for scalar sensors
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, n_nodes, 1)
        
        # GCN-style aggregation with adjacency
        x_agg = torch.matmul(adj, x)
        
        z = self.encoder(x_agg)
        recon = self.decoder(z)
        return recon.squeeze(-1)
    
    def get_reconstruction_error(self, x, adj):
        recon = self(x, adj)
        if x.dim() == 3:
            x = x.squeeze(-1)
        return torch.mean((recon - x) ** 2, dim=1)  # per-sample error


class GCNAEDetector:
    """Wrapper for GCN-AE drift detector."""
    
    def __init__(self, hidden_dim=32, epochs=50, lr=0.001):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = 'cpu'
    
    def fit(self, data: np.ndarray, edges: List[Tuple[int, int]]):
        n_sensors = data.shape[1]
        
        # Build adjacency matrix
        adj = np.zeros((n_sensors, n_sensors))
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Normalize adjacency (add self-loops)
        adj = adj + np.eye(n_sensors)
        D = np.diag(1.0 / np.sqrt(np.sum(adj, axis=1)))
        adj = D @ adj @ D
        
        self.adj = torch.FloatTensor(adj).to(self.device)
        
        # Create model
        self.model = GCNAE(n_sensors, in_dim=1, hidden_dim=self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Train
        X = torch.FloatTensor(data).to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon = self.model(X, self.adj)
            loss = F.mse_loss(recon, X)
            loss.backward()
            optimizer.step()
        
        # Compute baseline reconstruction error
        self.model.eval()
        with torch.no_grad():
            errors = self.model.get_reconstruction_error(X, self.adj).numpy()
        
        self.mu = np.mean(errors)
        self.sigma = np.std(errors) + 1e-6
        
        return self
    
    def score_batch(self, data: np.ndarray) -> np.ndarray:
        X = torch.FloatTensor(data).to(self.device)
        self.model.eval()
        with torch.no_grad():
            errors = self.model.get_reconstruction_error(X, self.adj).numpy()
        return (errors - self.mu) / self.sigma


# ============================================================================
# TRANSFORMER
# ============================================================================

class SimpleTransformer(nn.Module):
    """Simple Transformer for time series anomaly detection."""
    
    def __init__(self, n_sensors, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_sensors, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_sensors)
        self.n_sensors = n_sensors
    
    def forward(self, x):
        # x: (batch, seq_len, n_sensors) or (seq_len, n_sensors)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # add batch
        
        h = self.input_proj(x)
        h = self.transformer(h)
        out = self.output_proj(h)
        return out.squeeze(0)


class TransformerDetector:
    """Transformer-based detector using reconstruction error."""
    
    def __init__(self, window_size=10, d_model=32, epochs=50, lr=0.001):
        self.window_size = window_size
        self.d_model = d_model
        self.epochs = epochs
        self.lr = lr
        self.device = 'cpu'
    
    def fit(self, data: np.ndarray, edges=None):
        n_sensors = data.shape[1]
        
        # Create sliding windows
        windows = []
        for i in range(len(data) - self.window_size):
            windows.append(data[i:i+self.window_size])
        X = torch.FloatTensor(np.array(windows)).to(self.device)
        
        # Model
        self.model = SimpleTransformer(n_sensors, d_model=self.d_model).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Train on reconstruction
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon = self.model(X)
            loss = F.mse_loss(recon, X)
            loss.backward()
            optimizer.step()
        
        # Baseline error
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X)
            errors = torch.mean((recon - X) ** 2, dim=(1, 2)).numpy()
        
        self.mu = np.mean(errors)
        self.sigma = np.std(errors) + 1e-6
        self.n_sensors = n_sensors
        
        return self
    
    def score_batch(self, data: np.ndarray) -> np.ndarray:
        # Pad with zeros if needed
        n_samples = len(data)
        scores = np.zeros(n_samples)
        
        if n_samples < self.window_size:
            return scores
        
        self.model.eval()
        with torch.no_grad():
            for i in range(n_samples - self.window_size):
                window = torch.FloatTensor(data[i:i+self.window_size]).unsqueeze(0).to(self.device)
                recon = self.model(window)
                error = torch.mean((recon - window) ** 2).item()
                scores[i + self.window_size] = (error - self.mu) / self.sigma
        
        return scores


# ============================================================================
# SIMPLE NEURAL SHEAF (Bodnar-inspired)
# ============================================================================

class NeuralSheaf(nn.Module):
    """Simplified Neural Sheaf layer."""
    
    def __init__(self, n_nodes, in_dim=1, hidden_dim=16, out_dim=8):
        super().__init__()
        # Per-edge restriction maps
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.decoder = nn.Linear(out_dim, in_dim)
    
    def forward(self, x, edges):
        # x: (batch, n_nodes, in_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        batch_size, n_nodes, in_dim = x.shape
        
        # Node embeddings
        h = self.node_mlp(x)
        
        # Edge-based messages (simplified sheaf diffusion)
        messages = torch.zeros_like(h)
        for u, v in edges:
            edge_input = torch.cat([x[:, u], x[:, v]], dim=-1)
            edge_emb = self.edge_mlp(edge_input)
            messages[:, u] += edge_emb
            messages[:, v] += edge_emb
        
        h = h + messages
        recon = self.decoder(h)
        return recon.squeeze(-1)


class SheafDetector:
    """Neural Sheaf based detector."""
    
    def __init__(self, hidden_dim=16, epochs=50, lr=0.001):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = 'cpu'
    
    def fit(self, data: np.ndarray, edges: List[Tuple[int, int]]):
        n_sensors = data.shape[1]
        self.edges = edges
        
        self.model = NeuralSheaf(n_sensors, hidden_dim=self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        X = torch.FloatTensor(data).to(self.device)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon = self.model(X, edges)
            loss = F.mse_loss(recon, X)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X, edges)
            errors = torch.mean((recon - X) ** 2, dim=1).numpy()
        
        self.mu = np.mean(errors)
        self.sigma = np.std(errors) + 1e-6
        
        return self
    
    def score_batch(self, data: np.ndarray) -> np.ndarray:
        X = torch.FloatTensor(data).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X, self.edges)
            errors = torch.mean((recon - X) ** 2, dim=1).numpy()
        return (errors - self.mu) / self.sigma


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Quick test
    print("Testing DL Baselines")
    print("=" * 50)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_sensors = 20
    n_samples = 500
    
    # Generate data
    data = np.random.randn(n_samples, n_sensors)
    edges = [(i, i+1) for i in range(n_sensors-1)]
    
    print(f"Data: {n_samples} samples, {n_sensors} sensors, {len(edges)} edges")
    
    # Test each
    for name, cls in [('GCN-AE', GCNAEDetector), ('Transformer', TransformerDetector), ('Sheaf', SheafDetector)]:
        print(f"\n{name}:")
        det = cls(epochs=20)
        det.fit(data[:400], edges)
        scores = det.score_batch(data[400:])
        print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")
    
    print("\nDone!")
