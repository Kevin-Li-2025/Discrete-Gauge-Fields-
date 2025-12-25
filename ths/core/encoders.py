"""
Encoders for mapping real-world data to hypervectors.

These encoders transform various data types (continuous, categorical, sequences)
into the binary hypervector space H^D while preserving semantic relationships.

Key property: Similar inputs should produce similar hypervectors.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Union, Tuple
from ths.core.hypervector import (
    Hypervector, 
    bind, 
    bundle, 
    permute, 
    random_hypervector,
    random_orthogonal_set,
    similarity,
)


class RandomProjectionEncoder:
    """
    Encode continuous vectors via random binary projection.
    
    Uses locality-sensitive hashing (LSH) to map R^n to H^D.
    Preserves distances: similar vectors map to similar hypervectors
    (Johnson-Lindenstrauss-like guarantee).
    
    Algorithm:
    1. Project input through random matrix
    2. Apply sign function (threshold at 0)
    3. Pack into binary hypervector
    
    Attributes:
        input_dim: Dimension of input vectors
        output_dim: Dimension of output hypervectors
        projection_matrix: Random projection matrix
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int = 10000,
        seed: Optional[int] = None
    ):
        """
        Initialize the encoder.
        
        Args:
            input_dim: Dimension of input continuous vectors
            output_dim: Dimension of output hypervectors
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        
        rng = np.random.default_rng(seed)
        # Sparse random projection for efficiency
        # Using {-1, 0, +1} with probabilities {1/6, 2/3, 1/6}
        self.projection_matrix = rng.choice(
            [-1, 0, 0, 0, 0, 1],  # 2/3 zeros, 1/6 each for ±1
            size=(output_dim, input_dim)
        ).astype(np.float32)
        
        # Normalize rows for stable scaling
        norms = np.linalg.norm(self.projection_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.projection_matrix /= norms
        
    def encode(self, x: np.ndarray) -> Hypervector:
        """
        Encode a continuous vector to a hypervector.
        
        Args:
            x: Input vector of shape (input_dim,) or (batch, input_dim)
            
        Returns:
            Binary hypervector(s)
        """
        x = np.asarray(x, dtype=np.float32)
        
        if x.ndim == 1:
            # Single vector
            projection = self.projection_matrix @ x
            bits = (projection > 0).astype(np.uint8)
            return Hypervector.from_bits(bits)
        else:
            # Batch encoding
            projections = x @ self.projection_matrix.T
            return [
                Hypervector.from_bits((proj > 0).astype(np.uint8))
                for proj in projections
            ]
    
    def encode_batch(self, X: np.ndarray) -> List[Hypervector]:
        """Encode a batch of vectors."""
        X = np.asarray(X, dtype=np.float32)
        projections = X @ self.projection_matrix.T
        return [
            Hypervector.from_bits((proj > 0).astype(np.uint8))
            for proj in projections
        ]


class IDLevelEncoder:
    """
    Encode scalar values using level hypervectors with interpolation.
    
    Creates a set of 'level' hypervectors spanning [min_val, max_val].
    Intermediate values are encoded by interpolating (bundling) between
    adjacent levels, preserving the ordering structure.
    
    Useful for: sensor readings, normalized features, ordinal data
    
    Properties:
    - encode(a) similar to encode(b) iff |a - b| is small
    - Supports periodic values (e.g., angles) via wrap-around
    """
    
    def __init__(
        self,
        dim: int = 10000,
        n_levels: int = 100,
        min_val: float = 0.0,
        max_val: float = 1.0,
        circular: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize the level encoder.
        
        Args:
            dim: Hypervector dimension
            n_levels: Number of discrete levels
            min_val: Minimum value in range
            max_val: Maximum value in range
            circular: Whether values wrap around (e.g., angles)
            seed: Random seed
        """
        self.dim = dim
        self.n_levels = n_levels
        self.min_val = min_val
        self.max_val = max_val
        self.circular = circular
        
        # Generate level hypervectors
        # Adjacent levels share ~90% of bits for smooth interpolation
        rng = np.random.default_rng(seed)
        
        # Start with random base
        base = Hypervector.random(dim, rng)
        self.levels = [base]
        
        # Flip ~5% of bits for each subsequent level
        flip_prob = 0.05
        current = base.to_bits()
        
        for _ in range(1, n_levels):
            flip_mask = rng.random(dim) < flip_prob
            current = current.copy()
            current[flip_mask] = 1 - current[flip_mask]
            self.levels.append(Hypervector.from_bits(current))
            
        if circular:
            # Blend last level back toward first for continuity
            self.levels[-1] = bundle([self.levels[-1], self.levels[0]])
    
    def encode(self, value: float) -> Hypervector:
        """
        Encode a scalar value.
        
        Args:
            value: Scalar in [min_val, max_val]
            
        Returns:
            Interpolated hypervector
        """
        # Normalize to [0, n_levels-1]
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        normalized = np.clip(normalized, 0, 1)
        level_pos = normalized * (self.n_levels - 1)
        
        # Get adjacent levels
        lower_idx = int(np.floor(level_pos))
        upper_idx = min(lower_idx + 1, self.n_levels - 1)
        
        if lower_idx == upper_idx:
            return self.levels[lower_idx].copy()
        
        # Interpolate via weighted bundle
        weight = level_pos - lower_idx
        # For binary vectors, we probabilistically choose bits
        lower_bits = self.levels[lower_idx].to_bits()
        upper_bits = self.levels[upper_idx].to_bits()
        
        # Probabilistic interpolation
        rng = np.random.default_rng()
        mask = rng.random(self.dim) < weight
        result_bits = np.where(mask, upper_bits, lower_bits)
        
        return Hypervector.from_bits(result_bits.astype(np.uint8))
    
    def encode_batch(self, values: np.ndarray) -> List[Hypervector]:
        """Encode multiple scalar values."""
        return [self.encode(v) for v in values]


class SequenceEncoder:
    """
    Encode ordered sequences using permutation and binding.
    
    Each position in the sequence is marked by rotating the hypervector,
    then all positions are bound together.
    
    sequence = bind(perm^0(item_0), perm^1(item_1), ..., perm^n(item_n))
    
    Useful for: time series windows, n-grams, ordered sets
    """
    
    def __init__(self, item_encoder: Union[RandomProjectionEncoder, IDLevelEncoder, None] = None):
        """
        Initialize sequence encoder.
        
        Args:
            item_encoder: Encoder for individual items (optional)
        """
        self.item_encoder = item_encoder
        
    def encode(self, items: List[Union[Hypervector, np.ndarray, float]]) -> Hypervector:
        """
        Encode an ordered sequence.
        
        Args:
            items: List of items (either hypervectors or values to encode)
            
        Returns:
            Bound sequence hypervector
        """
        if not items:
            raise ValueError("Cannot encode empty sequence")
        
        # Convert items to hypervectors if needed
        hvs = []
        for item in items:
            if isinstance(item, Hypervector):
                hvs.append(item)
            elif self.item_encoder is not None:
                hvs.append(self.item_encoder.encode(item))
            else:
                raise ValueError("Need item_encoder for non-hypervector items")
        
        # Apply positional permutation and bundle
        positioned = [permute(hv, i) for i, hv in enumerate(hvs)]
        return bundle(positioned)
    
    def encode_ngrams(self, items: List[Hypervector], n: int) -> List[Hypervector]:
        """
        Encode all n-grams from a sequence.
        
        Args:
            items: List of hypervectors
            n: N-gram size
            
        Returns:
            List of n-gram hypervectors
        """
        if len(items) < n:
            return []
        
        ngrams = []
        for i in range(len(items) - n + 1):
            ngram = self.encode(items[i:i+n])
            ngrams.append(ngram)
        return ngrams


class NGramEncoder:
    """
    Encode text as character/word n-gram hypervectors.
    
    Uses a codebook of random hypervectors for each unique token,
    then combines them using the sequence encoder.
    """
    
    def __init__(
        self,
        dim: int = 10000,
        n: int = 3,
        char_level: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize n-gram encoder.
        
        Args:
            dim: Hypervector dimension
            n: N-gram size
            char_level: If True, encode characters; else encode words
            seed: Random seed
        """
        self.dim = dim
        self.n = n
        self.char_level = char_level
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Lazy codebook - generate on demand
        self._codebook: Dict[str, Hypervector] = {}
        self._seq_encoder = SequenceEncoder()
        
    def _get_token_hv(self, token: str) -> Hypervector:
        """Get or create hypervector for a token."""
        if token not in self._codebook:
            self._codebook[token] = Hypervector.random(self.dim, self.rng)
        return self._codebook[token]
    
    def encode(self, text: str) -> Hypervector:
        """
        Encode text as a bundle of n-gram hypervectors.
        
        Args:
            text: Input string
            
        Returns:
            Text hypervector
        """
        if self.char_level:
            tokens = list(text)
        else:
            tokens = text.split()
            
        if len(tokens) < self.n:
            # Pad short sequences
            tokens = tokens + [''] * (self.n - len(tokens))
        
        # Get hypervectors for each token
        token_hvs = [self._get_token_hv(t) for t in tokens]
        
        # Generate all n-grams and bundle
        ngrams = self._seq_encoder.encode_ngrams(token_hvs, self.n)
        
        if not ngrams:
            return self._get_token_hv(text)  # Fallback for very short text
            
        return bundle(ngrams)


class MultiFeatureEncoder:
    """
    Encode multiple features into a single hypervector.
    
    Each feature gets its own encoder and role hypervector.
    The final encoding binds each feature with its role and bundles all.
    
    result = bundle(bind(role_1, encode(feat_1)), bind(role_2, encode(feat_2)), ...)
    """
    
    def __init__(
        self,
        feature_specs: Dict[str, dict],
        dim: int = 10000,
        seed: Optional[int] = None
    ):
        """
        Initialize multi-feature encoder.
        
        Args:
            feature_specs: Dict mapping feature names to encoder specs
                Example: {
                    'temperature': {'type': 'level', 'min': -20, 'max': 50},
                    'location': {'type': 'categorical', 'values': ['A', 'B', 'C']},
                    'signal': {'type': 'projection', 'input_dim': 32}
                }
            dim: Hypervector dimension
            seed: Random seed
        """
        self.dim = dim
        self.feature_names = list(feature_specs.keys())
        
        rng = np.random.default_rng(seed)
        
        # Create role hypervectors for each feature
        self.roles = {
            name: Hypervector.random(dim, rng)
            for name in self.feature_names
        }
        
        # Create encoders for each feature
        self.encoders: Dict[str, Union[IDLevelEncoder, RandomProjectionEncoder]] = {}
        self.categorical_codebooks: Dict[str, Dict[str, Hypervector]] = {}
        
        for name, spec in feature_specs.items():
            enc_type = spec.get('type', 'level')
            
            if enc_type == 'level':
                self.encoders[name] = IDLevelEncoder(
                    dim=dim,
                    min_val=spec.get('min', 0),
                    max_val=spec.get('max', 1),
                    seed=seed
                )
            elif enc_type == 'projection':
                self.encoders[name] = RandomProjectionEncoder(
                    input_dim=spec['input_dim'],
                    output_dim=dim,
                    seed=seed
                )
            elif enc_type == 'categorical':
                # Create codebook for categorical values
                values = spec['values']
                self.categorical_codebooks[name] = {
                    v: Hypervector.random(dim, rng)
                    for v in values
                }
    
    def encode(self, features: Dict[str, Union[float, np.ndarray, str]]) -> Hypervector:
        """
        Encode a feature dictionary.
        
        Args:
            features: Dict mapping feature names to values
            
        Returns:
            Combined hypervector
        """
        bound_features = []
        
        for name in self.feature_names:
            if name not in features:
                continue
                
            value = features[name]
            role = self.roles[name]
            
            if name in self.categorical_codebooks:
                # Categorical feature
                hv = self.categorical_codebooks[name].get(value)
                if hv is None:
                    raise ValueError(f"Unknown category '{value}' for feature '{name}'")
            else:
                # Use encoder
                hv = self.encoders[name].encode(value)
            
            bound_features.append(bind(role, hv))
        
        return bundle(bound_features)
