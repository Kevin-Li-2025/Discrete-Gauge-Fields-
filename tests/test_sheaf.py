"""Tests for cellular sheaf implementation."""

import pytest
import numpy as np
from ths.core.hypervector import Hypervector, random_hypervector, popcount, bind
from ths.sheaf.cellular_sheaf import CellularSheaf
from ths.sheaf.graph import Graph, build_knn_graph, build_temporal_graph


class TestGraph:
    """Test graph construction."""
    
    def test_from_edges(self):
        """Test graph creation from edge list."""
        edges = [(0, 1), (1, 2), (0, 2)]
        g = Graph.from_edges(3, edges)
        
        assert g.n_nodes == 3
        assert g.n_edges() == 3
        assert set(g.neighbors(0)) == {1, 2}
    
    def test_knn_graph(self):
        """Test k-NN graph construction."""
        data = np.random.randn(20, 5)
        g = build_knn_graph(data, k=3)
        
        assert g.n_nodes == 20
        # Each node should have at least k neighbors
        for i in range(20):
            assert g.degree(i) >= 3
    
    def test_temporal_graph(self):
        """Test temporal chain graph."""
        g = build_temporal_graph(10, window=2)
        
        assert g.n_nodes == 10
        # Node 0 connects to 1 and 2
        assert set(g.neighbors(0)) == {1, 2}
        # Node 5 connects to 3,4,6,7
        assert set(g.neighbors(5)) == {3, 4, 6, 7}


class TestCellularSheaf:
    """Test CellularSheaf class."""
    
    def test_create(self):
        """Test sheaf creation."""
        edges = [(0, 1), (1, 2)]
        g = Graph.from_edges(3, edges)
        sheaf = CellularSheaf.create(g, dim=1000)
        
        assert sheaf.dim == 1000
        assert sheaf.graph.n_nodes == 3
        # Should have context vectors for all incidences
        assert len(sheaf.context_vectors) == 4  # 2 edges * 2 endpoints
    
    def test_restriction_map(self):
        """Test restriction map applies binding."""
        g = Graph.from_edges(2, [(0, 1)])
        sheaf = CellularSheaf.create(g, dim=1000)
        
        x = random_hypervector(1000)
        edge_idx = 0
        
        rho = sheaf.restriction(0, edge_idx, x)
        
        # Should be x XOR context
        expected = bind(x, sheaf.context_vectors[(0, edge_idx)])
        assert rho == expected
    
    def test_coboundary_global_section(self):
        """Energy = 0 for consistent assignment."""
        g = Graph.from_edges(2, [(0, 1)])
        sheaf = CellularSheaf.create(g, dim=1000)
        
        # Create consistent assignment: x_1 = x_0 XOR C_{0->e} XOR C_{1->e}
        x0 = random_hypervector(1000)
        c0 = sheaf.context_vectors[(0, 0)]
        c1 = sheaf.context_vectors[(1, 0)]
        # For consistency: x0 XOR c0 = x1 XOR c1
        # So: x1 = x0 XOR c0 XOR c1
        x1 = bind(bind(x0, c0), c1)
        
        node_data = {0: x0, 1: x1}
        
        # Energy should be 0 (or very close due to tie-breaking)
        energy = sheaf.sheaf_energy(node_data)
        assert energy == 0
    
    def test_coboundary_inconsistent(self):
        """Energy > 0 when disagreement exists."""
        g = Graph.from_edges(2, [(0, 1)])
        sheaf = CellularSheaf.create(g, dim=1000)
        
        # Random assignment (inconsistent with high probability)
        x0 = random_hypervector(1000)
        x1 = random_hypervector(1000)
        
        node_data = {0: x0, 1: x1}
        energy = sheaf.sheaf_energy(node_data)
        
        # Should have significant energy
        assert energy > 0
        # Normalized should be around 0.5
        assert 0.3 < sheaf.normalized_energy(node_data) < 0.7
    
    def test_learn_contexts(self):
        """Test context learning from data."""
        g = Graph.from_edges(3, [(0, 1), (1, 2)])
        sheaf = CellularSheaf.create(g, dim=1000)
        
        # Create training data with consistent relationships
        training = []
        for _ in range(10):
            x0 = random_hypervector(1000)
            # Node 1 is always XOR of node 0 with fixed vector
            relationship = random_hypervector(1000)
            x1 = bind(x0, relationship)
            x2 = bind(x1, relationship)
            training.append({0: x0, 1: x1, 2: x2})
        
        sheaf.learn_contexts_from_data(training)
        
        assert sheaf.is_learned()
        
        # Energy on training-like data should be low
        test_sample = training[0]
        energy = sheaf.normalized_energy(test_sample)
        assert energy < 0.5  # Should be reasonably low after learning


class TestSheafEnergy:
    """Test sheaf energy computation."""
    
    def test_edge_energies(self):
        """Test per-edge energy computation."""
        g = Graph.from_edges(3, [(0, 1), (1, 2)])
        sheaf = CellularSheaf.create(g, dim=1000)
        
        node_data = {
            0: random_hypervector(1000),
            1: random_hypervector(1000),
            2: random_hypervector(1000),
        }
        
        edge_energies = sheaf.edge_energies(node_data)
        total = sheaf.sheaf_energy(node_data)
        
        # Sum of edge energies should equal total
        assert sum(edge_energies.values()) == total
    
    def test_frustrated_edges(self):
        """Test identifying frustrated edges."""
        g = Graph.from_edges(3, [(0, 1), (1, 2)])
        sheaf = CellularSheaf.create(g, dim=1000)
        
        # Make one edge consistent, one inconsistent
        x0 = random_hypervector(1000)
        c0_e0 = sheaf.context_vectors[(0, 0)]
        c1_e0 = sheaf.context_vectors[(1, 0)]
        x1 = bind(bind(x0, c0_e0), c1_e0)  # Consistent with edge 0
        x2 = random_hypervector(1000)  # Inconsistent with edge 1
        
        node_data = {0: x0, 1: x1, 2: x2}
        
        frustrated = sheaf.get_frustrated_edges(node_data, threshold=0.3)
        
        # Edge 1 should be frustrated
        edge_indices = [e[0] for e in frustrated]
        assert 1 in edge_indices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
