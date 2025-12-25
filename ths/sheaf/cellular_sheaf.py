"""
Cellular Sheaf implementation using Hyperdimensional Computing.

This module implements the core sheaf structure where:
- Stalks are hypervector spaces (H^D)
- Restriction maps are binding operations (XOR with context vectors)
- Sheaf energy measures global inconsistency via Hamming distance

The key insight is that cohomological computations reduce to bitwise logic.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from ths.core.hypervector import (
    Hypervector,
    bind,
    bundle,
    hamming_distance,
    popcount,
    random_hypervector,
    bundle_streaming,
    finalize_bundle,
)
from ths.sheaf.graph import Graph


@dataclass
class CellularSheaf:
    """
    A Cellular Sheaf on a graph with hypervector stalks.
    
    The sheaf assigns a hypervector space to each node and edge,
    with restriction maps implemented as binding with context vectors.
    
    Mathematical Structure:
    - Stalks: F(v) = F(e) = H^D (binary hypervectors)
    - Restriction maps: ρ_{v→e}(x) = x ⊗ C_{v→e}
    - Coboundary: δ⁰(x)_e = ρ_u(x_u) ⊕ ρ_v(x_v)
    - Energy: E(x) = Σ_e popcount(δ⁰(x)_e)
    
    Attributes:
        dim: Hypervector dimension D
        graph: The underlying graph structure
        context_vectors: Dict mapping (node, edge_idx) to context hypervector
    """
    dim: int
    graph: Graph
    context_vectors: Dict[Tuple[int, int], Hypervector] = field(default_factory=dict)
    _learned: bool = field(default=False)
    
    @classmethod
    def create(cls, graph: Graph, dim: int = 10000) -> CellularSheaf:
        """
        Create a new sheaf on a graph with random context vectors.
        
        Args:
            graph: The underlying graph
            dim: Hypervector dimension
            
        Returns:
            Initialized CellularSheaf
        """
        sheaf = cls(dim=dim, graph=graph, context_vectors={})
        sheaf._initialize_contexts()
        return sheaf
    
    def _initialize_contexts(self):
        """Initialize random context vectors for all incidences."""
        for edge_idx, (u, v) in enumerate(self.graph.edges):
            # Each endpoint gets a context vector
            self.context_vectors[(u, edge_idx)] = random_hypervector(self.dim)
            self.context_vectors[(v, edge_idx)] = random_hypervector(self.dim)
    
    def restriction(
        self, 
        node: int, 
        edge_idx: int, 
        x: Hypervector
    ) -> Hypervector:
        """
        Apply the restriction map ρ_{v→e}.
        
        Maps data from a node stalk to the edge stalk via binding
        with the context vector.
        
        ρ_{v→e}(x) = x ⊗ C_{v→e}
        
        Args:
            node: Source node
            edge_idx: Target edge index
            x: Input hypervector from node stalk
            
        Returns:
            Restricted hypervector in edge stalk
        """
        context = self.context_vectors.get((node, edge_idx))
        if context is None:
            raise ValueError(f"No context for node {node}, edge {edge_idx}")
        return bind(x, context)
    
    def coboundary_at_edge(
        self,
        edge_idx: int,
        node_data: Dict[int, Hypervector]
    ) -> Hypervector:
        """
        Compute the coboundary at a single edge.
        
        δ⁰(x)_e = ρ_{u→e}(x_u) ⊕ ρ_{v→e}(x_v)
        
        This measures the local disagreement between the two endpoints.
        
        Args:
            edge_idx: Edge index
            node_data: Dict mapping node id to its hypervector
            
        Returns:
            Coboundary hypervector at this edge
        """
        u, v = self.graph.edges[edge_idx]
        
        x_u = node_data[u]
        x_v = node_data[v]
        
        # Apply restrictions
        rho_u = self.restriction(u, edge_idx, x_u)
        rho_v = self.restriction(v, edge_idx, x_v)
        
        # Coboundary is XOR (difference in F_2)
        return bind(rho_u, rho_v)
    
    def coboundary(
        self,
        node_data: Dict[int, Hypervector]
    ) -> Dict[int, Hypervector]:
        """
        Compute the full coboundary δ⁰(x).
        
        Returns the disagreement hypervector at each edge.
        
        Args:
            node_data: Dict mapping node id to its hypervector
            
        Returns:
            Dict mapping edge index to coboundary hypervector
        """
        result = {}
        for edge_idx in range(self.graph.n_edges()):
            result[edge_idx] = self.coboundary_at_edge(edge_idx, node_data)
        return result
    
    def sheaf_energy(self, node_data: Dict[int, Hypervector]) -> int:
        """
        Compute the total sheaf energy.
        
        E(x) = ||δ⁰(x)||_H = Σ_e popcount(δ⁰(x)_e)
        
        This measures total global inconsistency. 
        E = 0 means x is a global section (perfect consistency).
        
        Args:
            node_data: Dict mapping node id to its hypervector
            
        Returns:
            Total Hamming weight of coboundary
        """
        total_energy = 0
        for edge_idx in range(self.graph.n_edges()):
            coboundary = self.coboundary_at_edge(edge_idx, node_data)
            total_energy += popcount(coboundary)
        return total_energy
    
    def edge_energies(
        self, 
        node_data: Dict[int, Hypervector]
    ) -> Dict[int, int]:
        """
        Compute energy contribution at each edge.
        
        Useful for identifying which relationships are violated.
        
        Args:
            node_data: Dict mapping node id to its hypervector
            
        Returns:
            Dict mapping edge index to its energy contribution
        """
        energies = {}
        for edge_idx in range(self.graph.n_edges()):
            coboundary = self.coboundary_at_edge(edge_idx, node_data)
            energies[edge_idx] = popcount(coboundary)
        return energies
    
    def normalized_energy(self, node_data: Dict[int, Hypervector]) -> float:
        """
        Compute normalized sheaf energy in [0, 1].
        
        Divides by maximum possible energy (dim * n_edges).
        
        Args:
            node_data: Dict mapping node id to its hypervector
            
        Returns:
            Normalized energy value
        """
        raw_energy = self.sheaf_energy(node_data)
        max_energy = self.dim * self.graph.n_edges()
        return raw_energy / max_energy if max_energy > 0 else 0.0
    
    def learn_contexts_from_data(
        self,
        training_data: List[Dict[int, Hypervector]]
    ) -> None:
        """
        Learn context vectors from training data.
        
        For each edge (u,v) and each training sample, we observe
        the relationship x_u and x_v. The context C_{v→e} is learned
        as the bundled difference: C = Bundle(x_u ⊕ x_v).
        
        This ensures that normal data has low sheaf energy.
        
        Simplification: We fix C_{u→e} = 0 (identity) and learn C_{v→e}.
        Then ρ_u(x_u) = x_u and ρ_v(x_v) = x_v ⊗ C_{v→e}.
        For consistency: x_u = x_v ⊗ C_{v→e}
        So: C_{v→e} = x_u ⊕ x_v (learned via bundling)
        
        Args:
            training_data: List of node assignments for training
        """
        n_samples = len(training_data)
        if n_samples == 0:
            return
        
        # Initialize accumulators for streaming bundle
        accumulators: Dict[int, np.ndarray] = {}
        for edge_idx, (u, v) in enumerate(self.graph.edges):
            accumulators[edge_idx] = np.zeros(self.dim, dtype=np.int32)
        
        # Accumulate relationships
        for sample in training_data:
            for edge_idx, (u, v) in enumerate(self.graph.edges):
                if u in sample and v in sample:
                    x_u = sample[u]
                    x_v = sample[v]
                    # Relationship vector
                    relationship = bind(x_u, x_v)
                    accumulators[edge_idx] = bundle_streaming(
                        accumulators[edge_idx], relationship, n_samples
                    )
        
        # Finalize bundles and set contexts
        # Fix C_{u→e} = 0 (identity), learn C_{v→e}
        for edge_idx, (u, v) in enumerate(self.graph.edges):
            # Identity context for first endpoint
            self.context_vectors[(u, edge_idx)] = Hypervector.zeros(self.dim)
            # Learned context for second endpoint
            self.context_vectors[(v, edge_idx)] = finalize_bundle(
                accumulators[edge_idx], n_samples
            )
        
        self._learned = True
    
    def is_learned(self) -> bool:
        """Check if contexts have been learned from data."""
        return self._learned
    
    def get_frustrated_edges(
        self,
        node_data: Dict[int, Hypervector],
        threshold: float = 0.4
    ) -> List[Tuple[int, int, int]]:
        """
        Get edges with energy above threshold.
        
        Useful for interpretability - shows which relationships broke.
        
        Args:
            node_data: Current node assignment
            threshold: Normalized energy threshold (0-1)
            
        Returns:
            List of (edge_idx, node_u, node_v) for frustrated edges
        """
        frustrated = []
        energy_threshold = int(threshold * self.dim)
        
        for edge_idx, (u, v) in enumerate(self.graph.edges):
            coboundary = self.coboundary_at_edge(edge_idx, node_data)
            energy = popcount(coboundary)
            if energy > energy_threshold:
                frustrated.append((edge_idx, u, v))
        
        return frustrated
    
    def diffuse(
        self,
        node_data: Dict[int, Hypervector],
        iterations: int = 10
    ) -> Dict[int, Hypervector]:
        """
        Sheaf diffusion: iteratively update nodes toward consensus.
        
        This is the hypervector analog of Laplacian diffusion.
        Each node updates based on bundling its neighbors' transformed values.
        
        Converges toward global sections (minimum energy states).
        
        Args:
            node_data: Initial node assignment
            iterations: Number of diffusion steps
            
        Returns:
            Diffused node assignment
        """
        current = {k: v.copy() for k, v in node_data.items()}
        
        for _ in range(iterations):
            new_data = {}
            
            for node in range(self.graph.n_nodes):
                neighbors_transformed = []
                
                for neighbor in self.graph.neighbors(node):
                    edge_idx = self.graph.get_edge_index(node, neighbor)
                    
                    # Get neighbor's value
                    x_neighbor = current.get(neighbor)
                    if x_neighbor is None:
                        continue
                    
                    # Transform to node's frame
                    # If C_{neighbor→e} = relationship, and C_{node→e} = 0
                    # Then to get neighbor in node's frame: x_neighbor ⊗ C_{neighbor→e}
                    c_neighbor = self.context_vectors.get((neighbor, edge_idx))
                    c_node = self.context_vectors.get((node, edge_idx))
                    
                    if c_neighbor is not None and c_node is not None:
                        # Transform: x_neighbor in edge frame, then to node frame
                        in_edge_frame = bind(x_neighbor, c_neighbor)
                        in_node_frame = bind(in_edge_frame, c_node)  # XOR is self-inverse
                        neighbors_transformed.append(in_node_frame)
                
                if neighbors_transformed:
                    # Include self with high weight
                    if node in current:
                        neighbors_transformed.append(current[node])
                    new_data[node] = bundle(neighbors_transformed)
                elif node in current:
                    new_data[node] = current[node].copy()
            
            current = new_data
        
        return current


def compute_cohomology_dimension(sheaf: CellularSheaf) -> Dict[str, int]:
    """
    Estimate cohomology dimensions via random sampling.
    
    Not exact, but gives insight into the sheaf structure.
    
    - dim H^0: Number of independent global sections
    - dim H^1: Number of independent "holes" in consistency
    
    Args:
        sheaf: The cellular sheaf
        
    Returns:
        Dict with 'H0' and 'H1' estimates
    """
    # H^0 is isomorphic to kernel of coboundary
    # For random contexts, typically H^0 = 1 (just constant sections)
    
    # H^1 = dim(C^1) - dim(im δ^0)
    # = n_edges * D - rank(δ^0)
    
    # These are rough estimates; exact computation is expensive
    n_edges = sheaf.graph.n_edges()
    n_nodes = sheaf.graph.n_nodes
    
    # Euler characteristic: χ = n_nodes - n_edges = dim H^0 - dim H^1
    euler = n_nodes - n_edges
    
    return {
        'H0_estimate': 1,  # Usually 1 for connected graphs
        'H1_estimate': max(0, 1 - euler),  # Based on Euler char
        'euler_characteristic': euler
    }
