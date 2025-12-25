#!/usr/bin/env python3
"""
Basic usage example for Topological Hypervector Sheaves.

Demonstrates:
1. Creating hypervectors and using core operations
2. Building a sheaf on a graph
3. Computing sheaf energy
"""

import numpy as np
from ths.core.hypervector import (
    Hypervector, 
    bind, 
    bundle, 
    permute,
    similarity,
    hamming_distance,
    random_hypervector,
)
from ths.core.encoders import RandomProjectionEncoder, IDLevelEncoder
from ths.sheaf.cellular_sheaf import CellularSheaf
from ths.sheaf.graph import Graph, build_knn_graph


def demo_hypervector_operations():
    """Demonstrate core HDC operations."""
    print("=" * 60)
    print("PART 1: Hypervector Operations")
    print("=" * 60)
    
    # Create random hypervectors
    dim = 10000
    a = random_hypervector(dim)
    b = random_hypervector(dim)
    
    print(f"\nCreated two random {dim}-dimensional hypervectors")
    print(f"  Vector A: {a}")
    print(f"  Vector B: {b}")
    
    # Similarity between random vectors (should be ~0)
    sim = similarity(a, b)
    print(f"\nSimilarity(A, B) = {sim:.4f}")
    print("  (Random vectors are nearly orthogonal)")
    
    # Binding operation
    c = bind(a, b)
    print(f"\nBinding (XOR):")
    print(f"  C = A ⊗ B")
    print(f"  Similarity(C, A) = {similarity(c, a):.4f}")
    print(f"  Similarity(C, B) = {similarity(c, b):.4f}")
    print("  (Bound vector is dissimilar to both inputs)")
    
    # Self-inverse property
    recovered = bind(a, c)
    print(f"\nSelf-inverse property:")
    print(f"  A ⊗ (A ⊗ B) = B")
    print(f"  Similarity(recovered, B) = {similarity(recovered, b):.4f}")
    
    # Bundling operation
    vectors = [random_hypervector(dim) for _ in range(5)]
    bundled = bundle(vectors)
    print(f"\nBundling (Majority Vote):")
    print(f"  Bundled 5 random vectors")
    for i, v in enumerate(vectors):
        print(f"  Similarity(bundled, v{i}) = {similarity(bundled, v):.4f}")
    print("  (Bundled vector is similar to all inputs)")
    
    # Permutation
    p = permute(a, 100)
    print(f"\nPermutation:")
    print(f"  P = permute(A, 100)")
    print(f"  Similarity(P, A) = {similarity(p, a):.4f}")
    print("  (Permuted vector is dissimilar to original)")


def demo_encoders():
    """Demonstrate data encoding to hypervectors."""
    print("\n" + "=" * 60)
    print("PART 2: Encoders")
    print("=" * 60)
    
    # Random projection encoder for continuous data
    encoder = RandomProjectionEncoder(input_dim=10, output_dim=10000, seed=42)
    
    x1 = np.random.randn(10)
    x2 = x1 + 0.1 * np.random.randn(10)  # Similar to x1
    x3 = np.random.randn(10)  # Different
    
    hv1 = encoder.encode(x1)
    hv2 = encoder.encode(x2)
    hv3 = encoder.encode(x3)
    
    print(f"\nRandom Projection Encoder:")
    print(f"  x1 and x2 are similar (small perturbation)")
    print(f"  x1 and x3 are different")
    print(f"  Similarity(hv1, hv2) = {similarity(hv1, hv2):.4f}")
    print(f"  Similarity(hv1, hv3) = {similarity(hv1, hv3):.4f}")
    
    # Level encoder for scalar values
    level_enc = IDLevelEncoder(dim=10000, min_val=0, max_val=100, seed=42)
    
    hv_10 = level_enc.encode(10)
    hv_15 = level_enc.encode(15)
    hv_90 = level_enc.encode(90)
    
    print(f"\nLevel Encoder:")
    print(f"  Encoding scalar values 10, 15, and 90")
    print(f"  Similarity(10, 15) = {similarity(hv_10, hv_15):.4f}")
    print(f"  Similarity(10, 90) = {similarity(hv_10, hv_90):.4f}")
    print("  (Similar values produce similar hypervectors)")


def demo_sheaf_energy():
    """Demonstrate sheaf construction and energy computation."""
    print("\n" + "=" * 60)
    print("PART 3: Cellular Sheaf and Energy")
    print("=" * 60)
    
    # Create a simple triangle graph
    edges = [(0, 1), (1, 2), (0, 2)]
    graph = Graph.from_edges(3, edges)
    
    print(f"\nCreated triangle graph with 3 nodes and 3 edges")
    
    # Create sheaf
    dim = 10000
    sheaf = CellularSheaf.create(graph, dim=dim)
    
    print(f"Created cellular sheaf with {dim}-dimensional stalks")
    
    # Assign random data to nodes (inconsistent)
    random_data = {
        0: random_hypervector(dim),
        1: random_hypervector(dim),
        2: random_hypervector(dim),
    }
    
    energy_random = sheaf.sheaf_energy(random_data)
    norm_energy = sheaf.normalized_energy(random_data)
    
    print(f"\nRandom node assignment (inconsistent):")
    print(f"  Raw energy = {energy_random}")
    print(f"  Normalized energy = {norm_energy:.4f}")
    
    # Create consistent assignment
    # For consistency: restriction maps should agree on edges
    x0 = random_hypervector(dim)
    c0_e0 = sheaf.context_vectors[(0, 0)]
    c1_e0 = sheaf.context_vectors[(1, 0)]
    x1 = bind(bind(x0, c0_e0), c1_e0)
    
    c1_e1 = sheaf.context_vectors[(1, 1)]
    c2_e1 = sheaf.context_vectors[(2, 1)]
    x2 = bind(bind(x1, c1_e1), c2_e1)
    
    consistent_data = {0: x0, 1: x1, 2: x2}
    
    energy_consistent = sheaf.sheaf_energy(consistent_data)
    
    print(f"\nConsistent node assignment (global section):")
    print(f"  Energy = {energy_consistent}")
    print("  (Zero energy means perfect consistency)")
    
    # Edge energies
    edge_energies = sheaf.edge_energies(random_data)
    print(f"\nPer-edge energy breakdown (random assignment):")
    for edge_idx, (u, v) in enumerate(graph.edges):
        print(f"  Edge ({u}, {v}): {edge_energies[edge_idx]}")


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# Topological Hypervector Sheaves - Basic Usage Demo")
    print("#" * 60)
    
    demo_hypervector_operations()
    demo_encoders()
    demo_sheaf_energy()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
