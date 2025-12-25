"""
Synthetic data generators for concept drift experiments.

Provides controlled drift scenarios for evaluating detection algorithms:
- RotatingGaussian: Gradual drift via rotating distribution
- SuddenDrift: Abrupt distribution shift
- GradualDrift: Slow, linear interpolation between concepts
- RecurringDrift: Periodic concept switching
- SensorNetworkSimulator: Multi-sensor spatial data with faults
"""

from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class StreamSample:
    """A sample from a data stream."""
    data: np.ndarray
    time: int
    concept: int  # Which concept generated this sample
    is_drift_point: bool  # True if drift just occurred


class RotatingGaussian:
    """
    Rotating 2D Gaussian distribution.
    
    A Gaussian cloud rotates around the origin over time,
    creating a gradual concept drift scenario.
    
    Useful for testing detection of continuous distribution shifts.
    
    Example:
        >>> gen = RotatingGaussian(n_samples=1000, drift_start=500)
        >>> for sample in gen:
        ...     x = sample.data
        ...     if sample.is_drift_point:
        ...         print(f"Drift at t={sample.time}")
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        drift_start: int = 500,
        rotation_speed: float = 0.02,
        center: Tuple[float, float] = (2.0, 0.0),
        std: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize the rotating Gaussian generator.
        
        Args:
            n_samples: Total number of samples to generate
            drift_start: Time step when rotation begins
            rotation_speed: Radians per time step
            center: Initial center of Gaussian
            std: Standard deviation
            seed: Random seed
        """
        self.n_samples = n_samples
        self.drift_start = drift_start
        self.rotation_speed = rotation_speed
        self.center = np.array(center)
        self.std = std
        self.rng = np.random.default_rng(seed)
        
    def __iter__(self) -> Iterator[StreamSample]:
        for t in range(self.n_samples):
            # Rotate center after drift starts
            if t >= self.drift_start:
                angle = (t - self.drift_start) * self.rotation_speed
                rotation = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])
                rotated_center = rotation @ self.center
                concept = 1
            else:
                rotated_center = self.center
                concept = 0
            
            # Sample from Gaussian
            sample = self.rng.normal(rotated_center, self.std)
            
            yield StreamSample(
                data=sample,
                time=t,
                concept=concept,
                is_drift_point=(t == self.drift_start)
            )
    
    def generate_array(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate all samples as arrays.
        
        Returns:
            Tuple of (data array, concept labels, drift point index)
        """
        samples = list(self)
        data = np.array([s.data for s in samples])
        concepts = np.array([s.concept for s in samples])
        return data, concepts, self.drift_start


class SuddenDrift:
    """
    Sudden/abrupt concept drift.
    
    Distribution shifts suddenly at drift_point from
    one Gaussian to a completely different one.
    
    The classic drift scenario for testing detection speed.
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        drift_point: int = 500,
        dim: int = 10,
        concept1_mean: Optional[np.ndarray] = None,
        concept2_mean: Optional[np.ndarray] = None,
        std: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize sudden drift generator.
        
        Args:
            n_samples: Total samples
            drift_point: When drift occurs
            dim: Data dimensionality
            concept1_mean: Mean of first concept (default: zeros)
            concept2_mean: Mean of second concept (default: ones)
            std: Standard deviation
            seed: Random seed
        """
        self.n_samples = n_samples
        self.drift_point = drift_point
        self.dim = dim
        self.std = std
        self.rng = np.random.default_rng(seed)
        
        self.concept1_mean = concept1_mean if concept1_mean is not None else np.zeros(dim)
        self.concept2_mean = concept2_mean if concept2_mean is not None else np.ones(dim) * 2
        
    def __iter__(self) -> Iterator[StreamSample]:
        for t in range(self.n_samples):
            if t < self.drift_point:
                mean = self.concept1_mean
                concept = 0
            else:
                mean = self.concept2_mean
                concept = 1
            
            sample = self.rng.normal(mean, self.std)
            
            yield StreamSample(
                data=sample,
                time=t,
                concept=concept,
                is_drift_point=(t == self.drift_point)
            )
    
    def generate_array(self) -> Tuple[np.ndarray, np.ndarray, int]:
        samples = list(self)
        data = np.array([s.data for s in samples])
        concepts = np.array([s.concept for s in samples])
        return data, concepts, self.drift_point


class GradualDrift:
    """
    Gradual concept drift via linear interpolation.
    
    During the drift window, samples are increasingly likely
    to come from the new concept.
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        drift_start: int = 400,
        drift_end: int = 600,
        dim: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize gradual drift generator.
        
        Args:
            n_samples: Total samples
            drift_start: When drift begins
            drift_end: When drift completes
            dim: Dimensionality
            seed: Random seed
        """
        self.n_samples = n_samples
        self.drift_start = drift_start
        self.drift_end = drift_end
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        
        self.mean1 = np.zeros(dim)
        self.mean2 = np.ones(dim) * 2
        
    def __iter__(self) -> Iterator[StreamSample]:
        for t in range(self.n_samples):
            if t < self.drift_start:
                prob_new = 0.0
            elif t > self.drift_end:
                prob_new = 1.0
            else:
                prob_new = (t - self.drift_start) / (self.drift_end - self.drift_start)
            
            # Probabilistically sample from old or new concept
            if self.rng.random() < prob_new:
                mean = self.mean2
                concept = 1
            else:
                mean = self.mean1
                concept = 0
            
            sample = self.rng.normal(mean, 1.0)
            
            yield StreamSample(
                data=sample,
                time=t,
                concept=concept,
                is_drift_point=(t == self.drift_start)
            )


class RecurringDrift:
    """
    Recurring/seasonal concept drift.
    
    Concepts cycle periodically, simulating seasonal patterns
    or alternating system states.
    """
    
    def __init__(
        self,
        n_samples: int = 2000,
        period: int = 400,
        n_concepts: int = 2,
        dim: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize recurring drift generator.
        
        Args:
            n_samples: Total samples
            period: Cycle period (samples per full cycle)
            n_concepts: Number of concepts to cycle through
            dim: Dimensionality
            seed: Random seed
        """
        self.n_samples = n_samples
        self.period = period
        self.n_concepts = n_concepts
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        
        # Generate random concept means
        self.concept_means = [
            self.rng.uniform(-2, 2, size=dim) 
            for _ in range(n_concepts)
        ]
        
    def __iter__(self) -> Iterator[StreamSample]:
        prev_concept = 0
        for t in range(self.n_samples):
            # Determine current concept based on cycle
            segment = (t % self.period) / (self.period / self.n_concepts)
            concept = int(segment) % self.n_concepts
            
            mean = self.concept_means[concept]
            sample = self.rng.normal(mean, 1.0)
            
            is_drift = (concept != prev_concept) and (t > 0)
            prev_concept = concept
            
            yield StreamSample(
                data=sample,
                time=t,
                concept=concept,
                is_drift_point=is_drift
            )


class SensorNetworkSimulator:
    """
    Simulates a network of sensors with spatial correlations.
    
    Sensors have local correlations, and faults or environmental
    changes cause drift in sensor behavior.
    """
    
    def __init__(
        self,
        n_sensors: int = 20,
        n_samples: int = 1000,
        adjacency: Optional[np.ndarray] = None,
        fault_sensors: List[int] = None,
        fault_start: int = 500,
        noise_std: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize sensor network simulator.
        
        Args:
            n_sensors: Number of sensors
            n_samples: Number of time steps
            adjacency: Sensor adjacency matrix (None = random)
            fault_sensors: Which sensors have faults (indices)
            fault_start: When faults begin
            noise_std: Sensor noise level
            seed: Random seed
        """
        self.n_sensors = n_sensors
        self.n_samples = n_samples
        self.fault_sensors = fault_sensors or []
        self.fault_start = fault_start
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        
        # Build adjacency matrix if not provided
        if adjacency is None:
            # Random geometric graph
            positions = self.rng.uniform(0, 1, size=(n_sensors, 2))
            distances = np.linalg.norm(
                positions[:, np.newaxis] - positions[np.newaxis, :],
                axis=2
            )
            threshold = 0.4
            self.adjacency = (distances < threshold).astype(float)
            np.fill_diagonal(self.adjacency, 0)
        else:
            self.adjacency = adjacency
        
        # Base sensor values (correlated based on adjacency)
        self.base_values = self.rng.uniform(-1, 1, size=n_sensors)
        
    def __iter__(self) -> Iterator[StreamSample]:
        for t in range(self.n_samples):
            # Generate correlated sensor readings
            values = self.base_values.copy()
            
            # Add smooth temporal variation
            temporal = 0.5 * np.sin(2 * np.pi * t / 100)
            values += temporal
            
            # Add fault (drift) to specified sensors
            if t >= self.fault_start:
                for sensor_id in self.fault_sensors:
                    values[sensor_id] += 3.0  # Large offset
            
            # Add noise
            values += self.rng.normal(0, self.noise_std, size=self.n_sensors)
            
            concept = 1 if t >= self.fault_start and self.fault_sensors else 0
            
            yield StreamSample(
                data=values,
                time=t,
                concept=concept,
                is_drift_point=(t == self.fault_start and len(self.fault_sensors) > 0)
            )
    
    def get_adjacency(self) -> np.ndarray:
        """Get sensor adjacency matrix for building graph."""
        return self.adjacency


def generate_benchmark_datasets() -> dict:
    """
    Generate standard benchmark datasets for evaluation.
    
    Returns:
        Dict mapping dataset name to (X, y, drift_point) tuples
    """
    datasets = {}
    
    # Rotating Gaussian
    gen = RotatingGaussian(n_samples=2000, drift_start=1000, seed=42)
    X, y, dp = gen.generate_array()
    datasets['rotating_gaussian'] = (X, y, dp)
    
    # Sudden drift
    gen = SuddenDrift(n_samples=2000, drift_point=1000, dim=10, seed=42)
    X, y, dp = gen.generate_array()
    datasets['sudden_drift'] = (X, y, dp)
    
    # Gradual drift
    gen = GradualDrift(n_samples=2000, drift_start=800, drift_end=1200, dim=10, seed=42)
    samples = list(gen)
    X = np.array([s.data for s in samples])
    y = np.array([s.concept for s in samples])
    datasets['gradual_drift'] = (X, y, 800)
    
    return datasets
