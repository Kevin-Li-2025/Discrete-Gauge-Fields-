"""
Hyperdimensional Computing (HDC) / Vector Symbolic Architecture (VSA) core operations.

This module implements the fundamental operations on binary hypervectors in Hamming space.
All operations use bitwise logic (XOR, AND, OR) for maximum efficiency on edge hardware.

Mathematical Foundation:
- Space: Boolean Hypercube H^D = {0, 1}^D where D is typically 10,000
- Binding: XOR operation (creates orthogonal composite)
- Bundling: Majority vote (creates similar composite)
- Permutation: Bit rotation (encodes sequence/order)
"""

from __future__ import annotations
import numpy as np
from typing import Union, List, Optional, Tuple
from functools import lru_cache


# Lookup table for fast popcount on bytes
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint32)


class Hypervector:
    """
    A binary hypervector in the Hamming cube H^D.
    
    Internally stored as a packed numpy array of uint8 (each byte = 8 bits).
    This enables efficient bitwise operations using numpy vectorization.
    
    Attributes:
        dim: The dimension D of the hypervector
        data: The packed binary data as uint8 array
        
    Example:
        >>> hv = Hypervector.random(10000)
        >>> print(hv.dim)  # 10000
        >>> print(len(hv.data))  # 1250 bytes
    """
    
    __slots__ = ('dim', 'data')
    
    def __init__(self, data: np.ndarray, dim: Optional[int] = None):
        """
        Initialize a hypervector from packed uint8 data.
        
        Args:
            data: Packed binary data as uint8 numpy array
            dim: Original dimension (needed if dim % 8 != 0)
        """
        if data.dtype != np.uint8:
            raise ValueError(f"Data must be uint8, got {data.dtype}")
        self.data = data
        self.dim = dim if dim is not None else len(data) * 8
        
    @classmethod
    def random(cls, dim: int, rng: Optional[np.random.Generator] = None) -> Hypervector:
        """
        Generate a random binary hypervector.
        
        Each bit is independently set to 0 or 1 with equal probability.
        Due to concentration of measure, random vectors are nearly orthogonal.
        
        Args:
            dim: Dimension of the hypervector
            rng: Optional numpy random generator for reproducibility
            
        Returns:
            A random Hypervector of the specified dimension
        """
        if rng is None:
            rng = np.random.default_rng()
        n_bytes = (dim + 7) // 8
        data = rng.integers(0, 256, size=n_bytes, dtype=np.uint8)
        return cls(data, dim)
    
    @classmethod
    def zeros(cls, dim: int) -> Hypervector:
        """Create an all-zeros hypervector (identity for XOR)."""
        n_bytes = (dim + 7) // 8
        return cls(np.zeros(n_bytes, dtype=np.uint8), dim)
    
    @classmethod
    def ones(cls, dim: int) -> Hypervector:
        """Create an all-ones hypervector."""
        n_bytes = (dim + 7) // 8
        return cls(np.full(n_bytes, 255, dtype=np.uint8), dim)
    
    @classmethod
    def from_bits(cls, bits: np.ndarray) -> Hypervector:
        """
        Create a hypervector from an unpacked bit array.
        
        Args:
            bits: 1D array of 0s and 1s
            
        Returns:
            Packed Hypervector
        """
        bits = np.asarray(bits, dtype=np.uint8)
        dim = len(bits)
        # Pad to multiple of 8
        padded_len = (dim + 7) // 8 * 8
        if padded_len > dim:
            bits = np.pad(bits, (0, padded_len - dim))
        # Pack bits into bytes
        data = np.packbits(bits)
        return cls(data, dim)
    
    def to_bits(self) -> np.ndarray:
        """Unpack to an array of individual bits."""
        bits = np.unpackbits(self.data)
        return bits[:self.dim]
    
    def copy(self) -> Hypervector:
        """Create a deep copy."""
        return Hypervector(self.data.copy(), self.dim)
    
    def __repr__(self) -> str:
        return f"Hypervector(dim={self.dim}, popcount={popcount(self)})"
    
    def __eq__(self, other: Hypervector) -> bool:
        if not isinstance(other, Hypervector):
            return False
        return self.dim == other.dim and np.array_equal(self.data, other.data)
    
    def __hash__(self) -> int:
        return hash((self.dim, self.data.tobytes()))
    
    def __xor__(self, other: Hypervector) -> Hypervector:
        """Bind operation via XOR."""
        return bind(self, other)
    
    def __or__(self, other: Hypervector) -> Hypervector:
        """Bitwise OR (used in some bundling variants)."""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Hypervector(np.bitwise_or(self.data, other.data), self.dim)
    
    def __and__(self, other: Hypervector) -> Hypervector:
        """Bitwise AND."""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Hypervector(np.bitwise_and(self.data, other.data), self.dim)
    
    def __invert__(self) -> Hypervector:
        """Bitwise NOT (flip all bits)."""
        return Hypervector(np.bitwise_not(self.data), self.dim)


def bind(a: Hypervector, b: Hypervector) -> Hypervector:
    """
    Binding operation using XOR.
    
    Properties:
    - Commutative: bind(a, b) = bind(b, a)
    - Self-inverse: bind(a, bind(a, b)) = b
    - Preserves distance: d(bind(a,c), bind(b,c)) = d(a, b)
    - Result is dissimilar to both inputs (orthogonal in expectation)
    
    This operation is used for:
    - Creating role-filler pairs (bind(role, filler))
    - Implementing sheaf restriction maps
    - Contextualizing data
    
    Args:
        a: First hypervector
        b: Second hypervector
        
    Returns:
        XOR of the two hypervectors
        
    Raises:
        ValueError: If dimensions don't match
    """
    if a.dim != b.dim:
        raise ValueError(f"Dimension mismatch: {a.dim} vs {b.dim}")
    return Hypervector(np.bitwise_xor(a.data, b.data), a.dim)


def bind_multiple(*vectors: Hypervector) -> Hypervector:
    """Bind multiple vectors together via chained XOR."""
    if not vectors:
        raise ValueError("Need at least one vector")
    result = vectors[0].copy()
    for v in vectors[1:]:
        result = bind(result, v)
    return result


def bundle(vectors: List[Hypervector], weights: Optional[np.ndarray] = None) -> Hypervector:
    """
    Bundling operation using majority vote.
    
    Creates a hypervector that is similar to all input vectors.
    This is the "learning" or "superposition" operation.
    
    Properties:
    - The result has high similarity to all inputs
    - Commutative and associative
    - With random tie-breaking for even counts
    
    Algorithm:
    1. For each bit position, count the 1s across all vectors
    2. If count > n/2: output 1, else output 0
    3. Ties broken randomly (or by threshold)
    
    Args:
        vectors: List of hypervectors to bundle
        weights: Optional weights for weighted bundling
        
    Returns:
        Majority vote hypervector
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if not vectors:
        raise ValueError("Need at least one vector to bundle")
    
    dim = vectors[0].dim
    if not all(v.dim == dim for v in vectors):
        raise ValueError("All vectors must have the same dimension")
    
    n = len(vectors)
    
    # Unpack all vectors to bits
    bits_matrix = np.array([v.to_bits() for v in vectors], dtype=np.int32)
    
    if weights is not None:
        weights = np.asarray(weights).reshape(-1, 1)
        bit_counts = np.sum(bits_matrix * weights, axis=0)
        threshold = np.sum(weights) / 2
    else:
        bit_counts = np.sum(bits_matrix, axis=0)
        threshold = n / 2
    
    # Majority vote with random tie-breaking
    result_bits = np.zeros(dim, dtype=np.uint8)
    result_bits[bit_counts > threshold] = 1
    
    # Random tie-breaking for exact ties
    ties = (bit_counts == threshold) & (n % 2 == 0)
    if np.any(ties):
        result_bits[ties] = np.random.randint(0, 2, size=np.sum(ties), dtype=np.uint8)
    
    return Hypervector.from_bits(result_bits)


def bundle_streaming(accumulator: np.ndarray, new_vector: Hypervector, count: int) -> np.ndarray:
    """
    Streaming bundle operation for online learning.
    
    Maintains a running sum of bits that can be thresholded later.
    More memory-efficient than storing all vectors.
    
    Args:
        accumulator: Running sum array (int32)
        new_vector: New vector to add
        count: Current count of vectors
        
    Returns:
        Updated accumulator
    """
    bits = new_vector.to_bits().astype(np.int32)
    return accumulator + bits


def finalize_bundle(accumulator: np.ndarray, count: int) -> Hypervector:
    """Convert streaming bundle accumulator to final hypervector."""
    threshold = count / 2
    bits = (accumulator > threshold).astype(np.uint8)
    # Random tie-breaking
    ties = accumulator == threshold
    if np.any(ties):
        bits[ties] = np.random.randint(0, 2, size=np.sum(ties), dtype=np.uint8)
    return Hypervector.from_bits(bits)


def permute(v: Hypervector, shifts: int = 1) -> Hypervector:
    """
    Permutation operation via circular bit shift.
    
    Used to encode sequence order or directed relationships.
    Permuting creates an orthogonal vector while being reversible.
    
    Properties:
    - permute(permute(v, k), -k) = v
    - similarity(v, permute(v, k)) ≈ 0 for k > 0
    
    Args:
        v: Input hypervector
        shifts: Number of positions to shift (positive = right)
        
    Returns:
        Circularly shifted hypervector
    """
    bits = v.to_bits()
    shifted = np.roll(bits, shifts)
    return Hypervector.from_bits(shifted)


def inverse_permute(v: Hypervector, shifts: int = 1) -> Hypervector:
    """Inverse of permute: shift in opposite direction."""
    return permute(v, -shifts)


def popcount(v: Hypervector) -> int:
    """
    Population count: number of 1-bits in the hypervector.
    
    Uses a lookup table for efficiency. This is the key operation
    for computing Hamming distance and sheaf energy.
    
    Hardware note: Modern CPUs have POPCNT instruction that can
    process 64 bits per cycle. This numpy version is a software
    fallback that's still very efficient.
    
    Args:
        v: Input hypervector
        
    Returns:
        Number of set bits
    """
    return int(np.sum(_POPCOUNT_TABLE[v.data]))


def hamming_distance(a: Hypervector, b: Hypervector) -> int:
    """
    Hamming distance: number of bit positions that differ.
    
    d_H(a, b) = popcount(a XOR b)
    
    This is the fundamental distance metric in Hamming space.
    
    Args:
        a: First hypervector
        b: Second hypervector
        
    Returns:
        Number of differing bits
    """
    if a.dim != b.dim:
        raise ValueError(f"Dimension mismatch: {a.dim} vs {b.dim}")
    xor_result = np.bitwise_xor(a.data, b.data)
    return int(np.sum(_POPCOUNT_TABLE[xor_result]))


def similarity(a: Hypervector, b: Hypervector) -> float:
    """
    Cosine-like similarity in Hamming space.
    
    sim(a, b) = 1 - 2 * d_H(a, b) / D
    
    Properties:
    - Range: [-1, 1]
    - sim(a, a) = 1
    - sim(a, ~a) = -1
    - E[sim(random, random)] ≈ 0
    
    Args:
        a: First hypervector
        b: Second hypervector
        
    Returns:
        Similarity value in [-1, 1]
    """
    d = hamming_distance(a, b)
    return 1.0 - 2.0 * d / a.dim


def normalized_hamming(a: Hypervector, b: Hypervector) -> float:
    """
    Normalized Hamming distance in [0, 1].
    
    Args:
        a: First hypervector
        b: Second hypervector
        
    Returns:
        Normalized distance (0 = identical, 1 = opposite)
    """
    return hamming_distance(a, b) / a.dim


def random_hypervector(dim: int, seed: Optional[int] = None) -> Hypervector:
    """
    Convenience function to generate a random hypervector.
    
    Args:
        dim: Dimension
        seed: Optional random seed
        
    Returns:
        Random Hypervector
    """
    rng = np.random.default_rng(seed)
    return Hypervector.random(dim, rng)


def random_orthogonal_set(dim: int, n: int, seed: Optional[int] = None) -> List[Hypervector]:
    """
    Generate a set of n nearly-orthogonal random hypervectors.
    
    In high dimensions, random vectors are nearly orthogonal with
    high probability (concentration of measure).
    
    Args:
        dim: Dimension of each hypervector
        n: Number of hypervectors to generate
        seed: Optional random seed
        
    Returns:
        List of random hypervectors
    """
    rng = np.random.default_rng(seed)
    return [Hypervector.random(dim, rng) for _ in range(n)]


def cleanup_memory(v: Hypervector, codebook: List[Hypervector]) -> Tuple[Hypervector, float]:
    """
    Associative memory cleanup: find closest codebook entry.
    
    This is the "recall" operation in HDC - given a noisy or
    composite vector, find the original symbol.
    
    Args:
        v: Query hypervector (possibly noisy)
        codebook: List of clean prototype hypervectors
        
    Returns:
        Tuple of (closest hypervector, similarity score)
    """
    best_sim = -2.0
    best_match = None
    
    for prototype in codebook:
        sim = similarity(v, prototype)
        if sim > best_sim:
            best_sim = sim
            best_match = prototype
            
    return best_match, best_sim


# Aliases for common operations
xor = bind
majority = bundle
rotate = permute
