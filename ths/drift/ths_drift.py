"""
THS-Drift: Topological Hypervector Sheaf Concept Drift Detector.

This is the main algorithm implementing the paper's contribution:
- Phase I: Learn the topological structure (sheaf context vectors) from reference data
- Phase II: Monitor streaming data for concept drift via sheaf energy

The detector flags drift when the cohomological energy exceeds statistical thresholds,
indicating that new data violates the learned topological constraints.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass, field
from collections import deque

from ths.core.hypervector import Hypervector, hamming_distance
from ths.core.encoders import RandomProjectionEncoder, IDLevelEncoder
from ths.sheaf.cellular_sheaf import CellularSheaf
from ths.sheaf.graph import Graph, build_knn_graph, build_temporal_graph


@dataclass
class DriftResult:
    """Result of drift detection check."""
    is_drift: bool
    confidence: float
    energy: float
    threshold: float
    message: str = ""


class THSDrift:
    """
    Topological Hypervector Sheaf Concept Drift Detector.
    
    Uses sheaf cohomology to detect when streaming data violates
    the topological structure learned from reference data.
    
    The key insight is that concept drift manifests as increased
    "topological frustration" - the sheaf energy spikes when new
    data fails to satisfy the learned consistency constraints.
    
    All computations use bitwise operations (XOR, popcount) for
    extreme efficiency on edge devices.
    
    Example:
        >>> detector = THSDrift(dim=10000, k=5)
        >>> detector.fit(X_train)  # Learn topological structure
        >>> for x in stream:
        ...     detector.update(x)
        ...     if detector.detect().is_drift:
        ...         print("Drift detected!")
    
    Attributes:
        dim: Hypervector dimension
        k: Number of neighbors in k-NN graph
        alpha: Sensitivity parameter (std deviations for threshold)
        window_size: Sliding window size for statistics
        persistence: Number of high-energy samples before confirming drift
    """
    
    def __init__(
        self,
        dim: int = 10000,
        k: int = 5,
        alpha: float = 3.0,
        window_size: int = 100,
        persistence: int = 5,
        graph_type: str = 'knn',
        encoder: str = 'projection',
        use_morse_reduction: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize the THS drift detector.
        
        Args:
            dim: Hypervector dimension (higher = more accurate, slower)
            k: Number of neighbors for k-NN graph
            alpha: Threshold sensitivity (number of std devs)
            window_size: Size of sliding window for statistics
            persistence: Consecutive high-energy samples to confirm drift
            graph_type: 'knn' or 'temporal'
            encoder: 'projection' or 'level'
            use_morse_reduction: Apply Discrete Morse reduction
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.k = k
        self.alpha = alpha
        self.window_size = window_size
        self.persistence = persistence
        self.graph_type = graph_type
        self.encoder_type = encoder
        self.use_morse_reduction = use_morse_reduction
        self.seed = seed
        
        # Will be initialized in fit()
        self.sheaf: Optional[CellularSheaf] = None
        self.encoder: Optional[Union[RandomProjectionEncoder, IDLevelEncoder]] = None
        self.reference_data: Optional[np.ndarray] = None
        self.reference_hvs: List[Hypervector] = []
        
        # Online monitoring state
        self.t: int = 0  # Current time step
        self.energy_history: deque = deque(maxlen=window_size)
        self.energy_mean: float = 0.0
        self.energy_var: float = 0.0
        self.high_energy_count: int = 0
        self.drift_detected: bool = False
        self._fitted: bool = False
        
    def fit(self, X: np.ndarray) -> THSDrift:
        """
        Phase I: Learn the topological structure from reference data.
        
        This is a single-pass learning procedure:
        1. Encode data points to hypervectors
        2. Build a graph (k-NN or temporal)
        3. Create sheaf with learned context vectors
        4. Compute baseline energy statistics
        
        Args:
            X: Reference data matrix of shape (n_samples, n_features)
            
        Returns:
            self (for method chaining)
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        self.reference_data = X
        
        # Initialize encoder
        if self.encoder_type == 'projection':
            self.encoder = RandomProjectionEncoder(
                input_dim=n_features,
                output_dim=self.dim,
                seed=self.seed
            )
        else:
            # For scalar data, use level encoder
            self.encoder = IDLevelEncoder(
                dim=self.dim,
                min_val=X.min(),
                max_val=X.max(),
                seed=self.seed
            )
        
        # Encode all reference data
        if self.encoder_type == 'projection':
            self.reference_hvs = self.encoder.encode_batch(X)
        else:
            self.reference_hvs = [self.encoder.encode(x[0]) for x in X]
        
        # Build graph
        if self.graph_type == 'knn':
            graph = build_knn_graph(X, k=self.k)
        else:
            graph = build_temporal_graph(n_samples, window=self.k)
        
        # Apply Morse reduction if requested
        if self.use_morse_reduction:
            from ths.sheaf.graph import morse_reduction
            graph = morse_reduction(graph)
        
        # Create sheaf
        self.sheaf = CellularSheaf.create(graph, dim=self.dim)
        
        # Learn context vectors from reference data
        # Convert to list of node data dicts
        training_samples = [
            {i: self.reference_hvs[i] for i in range(n_samples)}
        ]
        self.sheaf.learn_contexts_from_data(training_samples)
        
        # Compute baseline energy statistics
        self._compute_baseline_statistics()
        
        self._fitted = True
        return self
    
    def _compute_baseline_statistics(self):
        """Compute mean and variance of energy on reference data."""
        if self.sheaf is None:
            return
            
        n = len(self.reference_hvs)
        node_data = {i: self.reference_hvs[i] for i in range(n)}
        
        # Compute baseline energy using the full graph
        # We simulate variation by computing per-edge energies and sampling
        edge_energies = self.sheaf.edge_energies(node_data)
        
        if not edge_energies:
            self.energy_mean = 0.5
            self.energy_var = 0.01
            return
        
        # Convert to list of normalized edge energies
        edge_energy_list = [e / self.dim for e in edge_energies.values()]
        
        # Use bootstrap sampling to estimate variance
        energies = []
        n_edges = len(edge_energy_list)
        n_bootstrap = min(20, max(5, n_edges))
        
        for _ in range(n_bootstrap):
            # Sample subset of edges and compute mean energy
            sample_size = max(1, n_edges // 2)
            indices = np.random.choice(n_edges, size=sample_size, replace=False)
            sample_energy = np.mean([edge_energy_list[i] for i in indices])
            energies.append(sample_energy)
        
        # Also add the global normalized energy
        global_energy = self.sheaf.normalized_energy(node_data)
        energies.append(global_energy)
        
        if energies:
            self.energy_mean = np.mean(energies)
            self.energy_var = np.var(energies) + 1e-8  # Avoid zero variance
            
            # Initialize history with baseline
            self.energy_history = deque(energies[-self.window_size:], maxlen=self.window_size)
    
    def update(self, x: np.ndarray) -> float:
        """
        Phase II: Process a new sample and compute its energy.
        
        Encodes the new sample, places it on the graph, and computes
        the local sheaf energy. Updates running statistics.
        
        Args:
            x: New data point of shape (n_features,)
            
        Returns:
            Current (normalized) sheaf energy
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before update()")
        
        self.t += 1
        x = np.asarray(x)
        
        # Encode new sample
        if self.encoder_type == 'projection':
            x_hv = self.encoder.encode(x)
        else:
            x_hv = self.encoder.encode(x[0] if x.ndim > 0 else x)
        
        # Find nearest neighbors in reference data
        # (In practice, this would use an efficient index structure)
        similarities = [
            hamming_distance(x_hv, ref_hv) for ref_hv in self.reference_hvs
        ]
        nearest_indices = np.argsort(similarities)[:self.k + 1]
        
        # Create local node data: new sample + its neighbors
        local_data = {i: self.reference_hvs[i] for i in nearest_indices}
        new_node_id = len(self.reference_hvs)  # Virtual node for new sample
        
        # Compute local energy relative to neighbors
        # Use the edges incident to nearest neighbors
        total_energy = 0
        edge_count = 0
        
        for neighbor_idx in nearest_indices:
            # Get the learned relationship
            for other_idx in nearest_indices:
                if neighbor_idx != other_idx:
                    edge_idx = self.sheaf.graph.get_edge_index(neighbor_idx, other_idx)
                    if edge_idx >= 0 and (neighbor_idx, edge_idx) in self.sheaf.context_vectors:
                        # Compute what energy would be if new sample replaced neighbor
                        local_data_with_new = local_data.copy()
                        local_data_with_new[neighbor_idx] = x_hv
                        
                        cob = self.sheaf.coboundary_at_edge(edge_idx, local_data_with_new)
                        from ths.core.hypervector import popcount
                        total_energy += popcount(cob)
                        edge_count += 1
        
        # Normalize energy
        if edge_count > 0:
            normalized_energy = total_energy / (edge_count * self.dim)
        else:
            normalized_energy = 0.0
        
        # Update running statistics
        self.energy_history.append(normalized_energy)
        
        # Update mean and variance using Welford's algorithm
        if len(self.energy_history) >= 2:
            history_array = np.array(self.energy_history)
            self.energy_mean = np.mean(history_array)
            self.energy_var = np.var(history_array) + 1e-8
        
        return normalized_energy
    
    def detect(self) -> DriftResult:
        """
        Check if concept drift has occurred.
        
        Uses a statistical test on the sheaf energy:
        - If energy > mean + alpha * std, flag potential drift
        - If high energy persists for 'persistence' samples, confirm drift
        
        Returns:
            DriftResult with detection status and confidence
        """
        if not self.energy_history:
            return DriftResult(
                is_drift=False,
                confidence=0.0,
                energy=0.0,
                threshold=0.0,
                message="No data received yet"
            )
        
        current_energy = self.energy_history[-1]
        std = np.sqrt(self.energy_var)
        threshold = self.energy_mean + self.alpha * std
        
        # Compute confidence as z-score
        z_score = (current_energy - self.energy_mean) / std if std > 0 else 0
        confidence = min(1.0, max(0.0, z_score / self.alpha))
        
        # Check if current energy exceeds threshold
        if current_energy > threshold:
            self.high_energy_count += 1
        else:
            self.high_energy_count = max(0, self.high_energy_count - 1)
        
        # Confirm drift if high energy persists
        if self.high_energy_count >= self.persistence:
            self.drift_detected = True
            return DriftResult(
                is_drift=True,
                confidence=confidence,
                energy=current_energy,
                threshold=threshold,
                message=f"Drift confirmed at t={self.t} after {self.persistence} high-energy samples"
            )
        elif current_energy > threshold:
            return DriftResult(
                is_drift=False,
                confidence=confidence,
                energy=current_energy,
                threshold=threshold,
                message=f"Potential drift ({self.high_energy_count}/{self.persistence})"
            )
        else:
            return DriftResult(
                is_drift=False,
                confidence=confidence,
                energy=current_energy,
                threshold=threshold,
                message="Normal"
            )
    
    def reset(self):
        """Reset the detector state (keep learned sheaf)."""
        self.t = 0
        self.energy_history = deque(maxlen=self.window_size)
        self.high_energy_count = 0
        self.drift_detected = False
        
        # Recompute baseline statistics
        if self._fitted:
            self._compute_baseline_statistics()
    
    def get_energy_history(self) -> np.ndarray:
        """Get the energy time series."""
        return np.array(self.energy_history)
    
    def get_frustrated_edges(self, x: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Get the edges that are frustrated by the new sample.
        
        Useful for interpretability - shows which learned relationships
        are violated by the new data.
        
        Args:
            x: New data point
            
        Returns:
            List of (edge_idx, node_u, node_v) for frustrated edges
        """
        if not self._fitted or self.sheaf is None:
            return []
        
        # Encode and find neighbors
        if self.encoder_type == 'projection':
            x_hv = self.encoder.encode(x)
        else:
            x_hv = self.encoder.encode(x[0])
        
        similarities = [
            hamming_distance(x_hv, ref_hv) for ref_hv in self.reference_hvs
        ]
        nearest = np.argsort(similarities)[:self.k + 1]
        
        # Create node data
        node_data = {i: self.reference_hvs[i] for i in nearest}
        node_data[nearest[0]] = x_hv  # Replace nearest with new sample
        
        return self.sheaf.get_frustrated_edges(node_data, threshold=0.4)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current detector statistics."""
        return {
            'time_step': self.t,
            'energy_mean': self.energy_mean,
            'energy_std': np.sqrt(self.energy_var),
            'current_energy': self.energy_history[-1] if self.energy_history else 0.0,
            'threshold': self.energy_mean + self.alpha * np.sqrt(self.energy_var),
            'high_energy_count': self.high_energy_count,
            'drift_detected': self.drift_detected,
        }


class THSDriftBatch(THSDrift):
    """
    Batch variant of THS-Drift for network-wide monitoring.
    
    Instead of processing individual samples, this variant
    monitors the entire network state at each time step.
    
    Useful for sensor networks where all sensors report simultaneously.
    """
    
    def update_batch(self, X: np.ndarray) -> float:
        """
        Update with a batch representing the entire network state.
        
        Args:
            X: Data matrix of shape (n_nodes, n_features)
            
        Returns:
            Network-wide sheaf energy
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before update()")
        
        self.t += 1
        X = np.asarray(X)
        n_nodes = X.shape[0]
        
        # Encode all nodes
        if self.encoder_type == 'projection':
            hvs = self.encoder.encode_batch(X)
        else:
            hvs = [self.encoder.encode(x[0]) for x in X]
        
        # Create node data
        node_data = {i: hvs[i] for i in range(n_nodes)}
        
        # Compute global sheaf energy
        energy = self.sheaf.normalized_energy(node_data)
        
        # Update history
        self.energy_history.append(energy)
        
        # Update statistics
        if len(self.energy_history) >= 2:
            history_array = np.array(self.energy_history)
            self.energy_mean = np.mean(history_array)
            self.energy_var = np.var(history_array) + 1e-8
        
        return energy
