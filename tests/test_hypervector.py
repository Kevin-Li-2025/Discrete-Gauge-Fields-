"""Tests for core hypervector operations."""

import pytest
import numpy as np
from ths.core.hypervector import (
    Hypervector,
    bind,
    bundle,
    permute,
    popcount,
    hamming_distance,
    similarity,
    random_hypervector,
    random_orthogonal_set,
)


class TestHypervector:
    """Test Hypervector class."""
    
    def test_random_creation(self):
        """Test random hypervector generation."""
        hv = Hypervector.random(10000)
        assert hv.dim == 10000
        # Check roughly 50% bits are set
        pop = popcount(hv)
        assert 4000 < pop < 6000
    
    def test_zeros_ones(self):
        """Test zero and one vector creation."""
        zeros = Hypervector.zeros(1000)
        ones = Hypervector.ones(1000)
        
        assert popcount(zeros) == 0
        assert popcount(ones) == 1000
    
    def test_from_bits(self):
        """Test creation from bit array."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        hv = Hypervector.from_bits(bits)
        
        assert hv.dim == 8
        assert popcount(hv) == 4
        
        # Roundtrip
        recovered = hv.to_bits()
        np.testing.assert_array_equal(bits, recovered)
    
    def test_copy(self):
        """Test deep copy."""
        hv1 = Hypervector.random(1000)
        hv2 = hv1.copy()
        
        assert hv1 == hv2
        assert hv1.data is not hv2.data


class TestBind:
    """Test binding (XOR) operation."""
    
    def test_bind_xor(self):
        """Verify bind is XOR."""
        a = Hypervector.random(1000)
        b = Hypervector.random(1000)
        c = bind(a, b)
        
        # Manual XOR
        expected = np.bitwise_xor(a.data, b.data)
        np.testing.assert_array_equal(c.data, expected)
    
    def test_bind_self_inverse(self):
        """Verify bind(a, bind(a, b)) = b."""
        a = Hypervector.random(1000)
        b = Hypervector.random(1000)
        
        c = bind(a, b)
        recovered = bind(a, c)
        
        assert recovered == b
    
    def test_bind_commutative(self):
        """Verify bind(a, b) = bind(b, a)."""
        a = Hypervector.random(1000)
        b = Hypervector.random(1000)
        
        assert bind(a, b) == bind(b, a)
    
    def test_bind_preserves_distance(self):
        """Verify d(bind(a,c), bind(b,c)) = d(a, b)."""
        a = Hypervector.random(1000)
        b = Hypervector.random(1000)
        c = Hypervector.random(1000)
        
        d_original = hamming_distance(a, b)
        d_bound = hamming_distance(bind(a, c), bind(b, c))
        
        assert d_original == d_bound
    
    def test_bind_creates_orthogonal(self):
        """Result should be dissimilar to both inputs."""
        a = Hypervector.random(10000)
        b = Hypervector.random(10000)
        c = bind(a, b)
        
        # Similarity should be near 0
        assert abs(similarity(c, a)) < 0.1
        assert abs(similarity(c, b)) < 0.1


class TestBundle:
    """Test bundling (majority vote) operation."""
    
    def test_bundle_single(self):
        """Bundle of single vector returns that vector."""
        a = Hypervector.random(1000)
        result = bundle([a])
        
        # Should be identical (or very close)
        assert similarity(result, a) > 0.99
    
    def test_bundle_similarity(self):
        """Bundled vector similar to all inputs."""
        vectors = random_orthogonal_set(10000, 5)
        result = bundle(vectors)
        
        for v in vectors:
            sim = similarity(result, v)
            # Should have positive similarity
            assert sim > 0.2
    
    def test_bundle_majority(self):
        """Test majority vote property."""
        # Create vectors where position 0-99 are all 1, rest random
        dim = 1000
        bits1 = np.zeros(dim, dtype=np.uint8)
        bits2 = np.zeros(dim, dtype=np.uint8)
        bits3 = np.zeros(dim, dtype=np.uint8)
        
        # First 100 bits all set in all vectors
        bits1[:100] = 1
        bits2[:100] = 1
        bits3[:100] = 1
        
        hvs = [
            Hypervector.from_bits(bits1),
            Hypervector.from_bits(bits2),
            Hypervector.from_bits(bits3),
        ]
        
        result = bundle(hvs)
        result_bits = result.to_bits()
        
        # First 100 bits should all be 1
        np.testing.assert_array_equal(result_bits[:100], np.ones(100))


class TestPermute:
    """Test permutation operation."""
    
    def test_permute_reversible(self):
        """Verify permute(permute(v, k), -k) = v."""
        v = Hypervector.random(1000)
        
        shifted = permute(v, 7)
        recovered = permute(shifted, -7)
        
        assert recovered == v
    
    def test_permute_creates_orthogonal(self):
        """Permuted vector should be dissimilar to original."""
        v = Hypervector.random(10000)
        shifted = permute(v, 100)
        
        sim = similarity(v, shifted)
        assert abs(sim) < 0.1


class TestPopcount:
    """Test population count."""
    
    def test_popcount_accuracy(self):
        """Compare to naive implementation."""
        hv = Hypervector.random(10000)
        
        # Naive count
        naive = np.sum(hv.to_bits())
        
        assert popcount(hv) == naive
    
    def test_popcount_zeros(self):
        """Zero vector has popcount 0."""
        assert popcount(Hypervector.zeros(1000)) == 0
    
    def test_popcount_ones(self):
        """All-ones vector has popcount = dim."""
        assert popcount(Hypervector.ones(1000)) == 1000


class TestHammingDistance:
    """Test Hamming distance and similarity."""
    
    def test_distance_self_zero(self):
        """Distance to self is 0."""
        a = Hypervector.random(1000)
        assert hamming_distance(a, a) == 0
    
    def test_distance_opposite(self):
        """Distance to inverse is dim."""
        a = Hypervector.random(1000)
        b = ~a
        assert hamming_distance(a, b) == 1000
    
    def test_similarity_self_one(self):
        """Similarity to self is 1."""
        a = Hypervector.random(1000)
        assert similarity(a, a) == 1.0
    
    def test_similarity_opposite_minus_one(self):
        """Similarity to inverse is -1."""
        a = Hypervector.random(1000)
        b = ~a
        assert similarity(a, b) == -1.0
    
    def test_random_orthogonality(self):
        """Random vectors should have ~0 similarity."""
        results = []
        for _ in range(100):
            a = Hypervector.random(10000)
            b = Hypervector.random(10000)
            results.append(similarity(a, b))
        
        mean_sim = np.mean(results)
        std_sim = np.std(results)
        
        # Mean should be near 0
        assert abs(mean_sim) < 0.05
        # Std should be small (concentration of measure)
        assert std_sim < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
