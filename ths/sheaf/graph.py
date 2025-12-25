"""
Graph construction utilities for Topological Hypervector Sheaves.

Provides functions to build graphs from data that will serve as the
base space for the sheaf. Supports k-NN graphs from data points,
temporal chain graphs for time series, and fixed sensor network topologies.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import heapq


@dataclass
class Graph:
    """
    Simple graph representation for sheaf computations.
    
    Attributes:
        n_nodes: Number of nodes
        edges: List of (u, v) tuples
        edge_index: Dict mapping (u,v) to edge index
        adjacency: Dict mapping node to list of neighbors
    """
    n_nodes: int
    edges: List[Tuple[int, int]]
    edge_index: Dict[Tuple[int, int], int]
    adjacency: Dict[int, List[int]]
    
    @classmethod
    def from_edges(cls, n_nodes: int, edges: List[Tuple[int, int]]) -> Graph:
        """Create graph from edge list."""
        edge_index = {}
        adjacency = defaultdict(list)
        
        for i, (u, v) in enumerate(edges):
            # Store both directions for undirected
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            adjacency[u].append(v)
            adjacency[v].append(u)
            
        return cls(
            n_nodes=n_nodes,
            edges=edges,
            edge_index=dict(edge_index),
            adjacency=dict(adjacency)
        )
    
    def neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node."""
        return self.adjacency.get(node, [])
    
    def degree(self, node: int) -> int:
        """Get degree of a node."""
        return len(self.adjacency.get(node, []))
    
    def get_edge_index(self, u: int, v: int) -> int:
        """Get index of edge (u,v)."""
        return self.edge_index.get((u, v), -1)
    
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self.edges)


def build_knn_graph(
    data: np.ndarray,
    k: int = 5,
    metric: str = 'euclidean'
) -> Graph:
    """
    Build a k-nearest neighbors graph from data points.
    
    Each point becomes a node, and edges connect each point
    to its k nearest neighbors. The graph is made symmetric.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        k: Number of nearest neighbors
        metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        Graph with k-NN connectivity
    """
    n = len(data)
    
    if k >= n:
        k = n - 1
    
    # Compute pairwise distances
    if metric == 'euclidean':
        # Efficient squared Euclidean distance
        sq_norms = np.sum(data ** 2, axis=1)
        distances = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * data @ data.T
        distances = np.maximum(distances, 0)  # Numerical stability
    elif metric == 'cosine':
        # Cosine distance = 1 - cosine_similarity
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = data / norms
        similarities = normalized @ normalized.T
        distances = 1 - similarities
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Set diagonal to infinity to exclude self-loops
    np.fill_diagonal(distances, np.inf)
    
    # Find k nearest neighbors for each point
    edges_set: Set[Tuple[int, int]] = set()
    
    for i in range(n):
        # Get indices of k smallest distances
        neighbors = np.argpartition(distances[i], k)[:k]
        for j in neighbors:
            # Add edge in canonical order
            edge = (min(i, j), max(i, j))
            edges_set.add(edge)
    
    edges = sorted(list(edges_set))
    return Graph.from_edges(n, edges)


def build_temporal_graph(
    n_points: int,
    window: int = 1
) -> Graph:
    """
    Build a temporal chain graph for time series data.
    
    Creates a graph where each time point is connected to
    its neighbors within a sliding window.
    
    Args:
        n_points: Number of time points
        window: Connection window (connect t to t±window)
        
    Returns:
        Chain/path graph with windowed connectivity
    """
    edges = []
    
    for i in range(n_points):
        for offset in range(1, window + 1):
            j = i + offset
            if j < n_points:
                edges.append((i, j))
    
    return Graph.from_edges(n_points, edges)


def build_sensor_network(
    adjacency: np.ndarray
) -> Graph:
    """
    Build graph from a fixed adjacency matrix.
    
    Useful for known sensor network topologies where
    the connectivity is predefined.
    
    Args:
        adjacency: Binary adjacency matrix (n x n)
        
    Returns:
        Graph with given connectivity
    """
    n = adjacency.shape[0]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                edges.append((i, j))
    
    return Graph.from_edges(n, edges)


def build_grid_graph(rows: int, cols: int) -> Graph:
    """
    Build a 2D grid graph (4-connected).
    
    Useful for image-like data or spatial sensor arrays.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        Grid graph
    """
    n = rows * cols
    edges = []
    
    def node_id(r, c):
        return r * cols + c
    
    for r in range(rows):
        for c in range(cols):
            current = node_id(r, c)
            # Right neighbor
            if c + 1 < cols:
                edges.append((current, node_id(r, c + 1)))
            # Down neighbor
            if r + 1 < rows:
                edges.append((current, node_id(r + 1, c)))
    
    return Graph.from_edges(n, edges)


def morse_reduction(graph: Graph) -> Graph:
    """
    Apply Discrete Morse Theory reduction to simplify graph.
    
    Reduces computation by collapsing edges that don't affect
    the cohomology. Only 'critical' edges remain.
    
    Implementation: Greedy algorithm that pairs nodes with edges
    in a way that creates a gradient. Unpaired edges are critical.
    
    This is a simplified heuristic version. Full DMT would involve
    computing the Morse complex properly.
    
    Args:
        graph: Input graph
        
    Returns:
        Reduced graph with only critical edges
    """
    n = graph.n_nodes
    n_edges = graph.n_edges()
    
    # Track which nodes/edges are paired
    node_paired = [False] * n
    edge_paired = [False] * n_edges
    
    # Priority queue: (degree, node_id)
    # Process low-degree nodes first
    pq = [(graph.degree(i), i) for i in range(n)]
    heapq.heapify(pq)
    
    # Greedy pairing: pair each unpaired node with one incident edge
    while pq:
        _, node = heapq.heappop(pq)
        
        if node_paired[node]:
            continue
        
        # Find an unpaired incident edge
        for neighbor in graph.neighbors(node):
            edge_idx = graph.get_edge_index(node, neighbor)
            if not edge_paired[edge_idx] and not node_paired[neighbor]:
                # Pair this node with this edge
                node_paired[node] = True
                edge_paired[edge_idx] = True
                break
    
    # Critical edges are unpaired edges
    critical_edges = [
        graph.edges[i] 
        for i in range(n_edges) 
        if not edge_paired[i]
    ]
    
    if not critical_edges:
        # No critical edges - graph is tree-like
        # Return a spanning tree edge as critical
        if graph.edges:
            critical_edges = [graph.edges[0]]
    
    return Graph.from_edges(n, critical_edges)


def compute_spanning_tree(graph: Graph, root: int = 0) -> Graph:
    """
    Compute a spanning tree using BFS.
    
    Args:
        graph: Input graph
        root: Root node for BFS
        
    Returns:
        Spanning tree as a Graph
    """
    n = graph.n_nodes
    visited = [False] * n
    tree_edges = []
    
    queue = [root]
    visited[root] = True
    
    while queue:
        node = queue.pop(0)
        for neighbor in graph.neighbors(node):
            if not visited[neighbor]:
                visited[neighbor] = True
                tree_edges.append((min(node, neighbor), max(node, neighbor)))
                queue.append(neighbor)
    
    return Graph.from_edges(n, tree_edges)


def get_cycle_edges(graph: Graph) -> List[Tuple[int, int]]:
    """
    Find edges that form cycles (non-tree edges).
    
    These are the edges that contribute to H^1 cohomology.
    
    Args:
        graph: Input graph
        
    Returns:
        List of cycle-forming edges
    """
    spanning_tree = compute_spanning_tree(graph)
    tree_edge_set = set(spanning_tree.edges)
    
    cycle_edges = [
        edge for edge in graph.edges
        if edge not in tree_edge_set
    ]
    
    return cycle_edges
