"""
Baseline drift detectors for comparison.

Implements standard approaches:
- Deep Autoencoder (reconstruction error)
- PCA (projection error)
- ADWIN (adaptive windowing)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
from collections import deque
from dataclasses import dataclass

# Conditional imports for optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DriftResult:
    """Result of drift detection check."""
    is_drift: bool
    confidence: float
    metric_value: float
    threshold: float
    message: str = ""


class PCADrift:
    """
    PCA-based concept drift detector.
    
    Monitors the reconstruction error when projecting data onto
    the top k principal components learned from reference data.
    
    Drift is signaled when the projection error exceeds a threshold.
    """
    
    def __init__(
        self,
        n_components: int = 10,
        alpha: float = 3.0,
        window_size: int = 100
    ):
        """
        Initialize PCA drift detector.
        
        Args:
            n_components: Number of principal components to keep
            alpha: Sensitivity (std deviations for threshold)
            window_size: Sliding window for statistics
        """
        self.n_components = n_components
        self.alpha = alpha
        self.window_size = window_size
        
        self.mean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None
        self.error_mean: float = 0.0
        self.error_var: float = 1e-8
        self.error_history: deque = deque(maxlen=window_size)
        self._fitted: bool = False
        
    def fit(self, X: np.ndarray) -> PCADrift:
        """
        Fit PCA on reference data.
        
        Args:
            X: Reference data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Keep top k components
        k = min(self.n_components, n_features)
        self.components = Vt[:k].T  # (n_features, k)
        
        # Compute baseline reconstruction errors
        errors = []
        for x in X:
            error = self._reconstruction_error(x)
            errors.append(error)
        
        self.error_mean = np.mean(errors)
        self.error_var = np.var(errors) + 1e-8
        self.error_history = deque(errors[-self.window_size:], maxlen=self.window_size)
        
        self._fitted = True
        return self
    
    def _reconstruction_error(self, x: np.ndarray) -> float:
        """Compute reconstruction error for a single sample."""
        x_centered = x - self.mean
        projection = x_centered @ self.components  # Project onto components
        reconstruction = projection @ self.components.T  # Reconstruct
        error = np.linalg.norm(x_centered - reconstruction)
        return error
    
    def update(self, x: np.ndarray) -> float:
        """
        Process a new sample.
        
        Args:
            x: New data point
            
        Returns:
            Reconstruction error
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        
        error = self._reconstruction_error(x)
        self.error_history.append(error)
        
        # Update statistics
        if len(self.error_history) >= 2:
            arr = np.array(self.error_history)
            self.error_mean = np.mean(arr)
            self.error_var = np.var(arr) + 1e-8
        
        return error
    
    def detect(self) -> DriftResult:
        """Check for drift based on reconstruction error."""
        if not self.error_history:
            return DriftResult(False, 0.0, 0.0, 0.0, "No data")
        
        current = self.error_history[-1]
        std = np.sqrt(self.error_var)
        threshold = self.error_mean + self.alpha * std
        
        z_score = (current - self.error_mean) / std if std > 0 else 0
        confidence = min(1.0, max(0.0, z_score / self.alpha))
        
        is_drift = current > threshold
        
        return DriftResult(
            is_drift=is_drift,
            confidence=confidence,
            metric_value=current,
            threshold=threshold,
            message="Drift" if is_drift else "Normal"
        )


class ADWINDrift:
    """
    ADWIN (ADaptive WINdowing) drift detector.
    
    Maintains a window of recent values and detects when the
    mean changes significantly. Uses a statistical test to
    find the optimal window cut point.
    
    Reference: Bifet, A., & Gavalda, R. (2007). Learning from
    time-changing data with adaptive windowing.
    """
    
    def __init__(
        self,
        delta: float = 0.002,
        max_window: int = 1000
    ):
        """
        Initialize ADWIN detector.
        
        Args:
            delta: Confidence parameter for change detection
            max_window: Maximum window size
        """
        self.delta = delta
        self.max_window = max_window
        
        self.window: deque = deque(maxlen=max_window)
        self.t: int = 0
        self._last_drift: int = 0
        
    def fit(self, X: np.ndarray) -> ADWINDrift:
        """
        Initialize with reference data.
        
        For ADWIN, we just fill the window with mean values.
        """
        values = np.mean(X, axis=1) if X.ndim > 1 else X
        for v in values:
            self.window.append(v)
        return self
    
    def update(self, x: np.ndarray) -> float:
        """
        Add new value to window.
        
        Args:
            x: New data point (or scalar)
            
        Returns:
            Current value
        """
        self.t += 1
        value = np.mean(x) if hasattr(x, '__len__') else x
        self.window.append(value)
        
        # Trim window if drift detected
        self._detect_and_trim()
        
        return value
    
    def _detect_and_trim(self):
        """Check for drift and trim window if found."""
        n = len(self.window)
        if n < 10:
            return
        
        window_array = np.array(self.window)
        
        # Try different split points
        for i in range(5, n - 5):
            n0 = i
            n1 = n - i
            
            mean0 = np.mean(window_array[:i])
            mean1 = np.mean(window_array[i:])
            
            # Hoeffding bound
            m = 1.0 / ((1.0 / n0) + (1.0 / n1))
            eps = np.sqrt((1.0 / (2 * m)) * np.log(4.0 / self.delta))
            
            if abs(mean0 - mean1) > eps:
                # Drift detected - trim window
                self.window = deque(list(self.window)[i:], maxlen=self.max_window)
                self._last_drift = self.t
                return
    
    def detect(self) -> DriftResult:
        """Check if drift was recently detected."""
        is_drift = (self.t - self._last_drift) < 5
        
        window_array = np.array(self.window)
        mean = np.mean(window_array) if len(window_array) > 0 else 0
        
        return DriftResult(
            is_drift=is_drift,
            confidence=1.0 if is_drift else 0.0,
            metric_value=mean,
            threshold=0.0,
            message="Drift" if is_drift else "Normal"
        )


class AutoencoderDrift:
    """
    Deep Autoencoder drift detector.
    
    Trains an autoencoder on reference data and monitors
    reconstruction error for drift.
    
    Requires PyTorch.
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [64, 32, 64],
        epochs: int = 100,
        lr: float = 0.001,
        alpha: float = 3.0,
        window_size: int = 100,
        device: str = 'cpu'
    ):
        """
        Initialize autoencoder drift detector.
        
        Args:
            hidden_dims: Hidden layer dimensions (e.g., [64, 32, 64])
            epochs: Training epochs
            lr: Learning rate
            alpha: Sensitivity (std devs for threshold)
            window_size: Sliding window for statistics
            device: PyTorch device ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for AutoencoderDrift. "
                            "Install with: pip install torch")
        
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.window_size = window_size
        self.device = device
        
        self.model: Optional[nn.Module] = None
        self.error_mean: float = 0.0
        self.error_var: float = 1e-8
        self.error_history: deque = deque(maxlen=window_size)
        self._fitted: bool = False
    
    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the autoencoder model."""
        layers = []
        dims = [input_dim] + self.hidden_dims + [input_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on output
                layers.append(nn.ReLU())
        
        return nn.Sequential(*layers).to(self.device)
    
    def fit(self, X: np.ndarray) -> AutoencoderDrift:
        """
        Train autoencoder on reference data.
        
        Args:
            X: Reference data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        
        # Build model
        self.model = self._build_model(n_features)
        
        # Convert to tensor
        X_tensor = torch.tensor(X, device=self.device)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, X_tensor)
            loss.backward()
            optimizer.step()
        
        # Compute baseline errors
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor).cpu().numpy()
            errors = np.mean((X - reconstructed) ** 2, axis=1)
        
        self.error_mean = np.mean(errors)
        self.error_var = np.var(errors) + 1e-8
        self.error_history = deque(errors[-self.window_size:], maxlen=self.window_size)
        
        self._fitted = True
        return self
    
    def update(self, x: np.ndarray) -> float:
        """
        Compute reconstruction error for new sample.
        
        Args:
            x: New data point
            
        Returns:
            Reconstruction error (MSE)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, device=self.device)
            reconstructed = self.model(x_tensor).cpu().numpy()
        
        error = float(np.mean((x - reconstructed) ** 2))
        self.error_history.append(error)
        
        # Update statistics
        if len(self.error_history) >= 2:
            arr = np.array(self.error_history)
            self.error_mean = np.mean(arr)
            self.error_var = np.var(arr) + 1e-8
        
        return error
    
    def detect(self) -> DriftResult:
        """Check for drift based on reconstruction error."""
        if not self.error_history:
            return DriftResult(False, 0.0, 0.0, 0.0, "No data")
        
        current = self.error_history[-1]
        std = np.sqrt(self.error_var)
        threshold = self.error_mean + self.alpha * std
        
        z_score = (current - self.error_mean) / std if std > 0 else 0
        confidence = min(1.0, max(0.0, z_score / self.alpha))
        
        is_drift = current > threshold
        
        return DriftResult(
            is_drift=is_drift,
            confidence=confidence,
            metric_value=current,
            threshold=threshold,
            message="Drift" if is_drift else "Normal"
        )
