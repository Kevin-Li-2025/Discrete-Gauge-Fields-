"""
Visualization utilities for THS drift detection.

Provides plotting functions for:
- Energy traces over time
- Drift detection results
- Algorithm comparisons
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_energy_trace(
    energies: np.ndarray,
    threshold: Optional[float] = None,
    drift_points: Optional[List[int]] = None,
    title: str = "Sheaf Energy Over Time",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot energy trace with optional threshold and drift markers.
    
    Args:
        energies: Array of energy values over time
        threshold: Detection threshold (horizontal line)
        drift_points: Indices where drift occurs (vertical lines)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    time = np.arange(len(energies))
    ax.plot(time, energies, 'b-', linewidth=1.0, label='Energy')
    
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', 
                   linewidth=1.5, label=f'Threshold ({threshold:.3f})')
    
    if drift_points:
        for i, dp in enumerate(drift_points):
            label = 'True Drift' if i == 0 else None
            ax.axvline(x=dp, color='green', linestyle=':', 
                       linewidth=2.0, alpha=0.7, label=label)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Cohomological Energy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_drift_detection(
    energies: np.ndarray,
    detections: List[int],
    true_drift: int,
    threshold: float,
    title: str = "Drift Detection Results",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comprehensive drift detection visualization.
    
    Shows energy trace, threshold, true drift, and detections.
    
    Args:
        energies: Energy values
        detections: Detected drift indices
        true_drift: True drift point
        threshold: Detection threshold
        title: Plot title
        figsize: Figure size
        save_path: Save path
        
    Returns:
        Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    time = np.arange(len(energies))
    
    # Top: Energy trace
    ax1 = axes[0]
    ax1.fill_between(time, energies, alpha=0.3, color='blue')
    ax1.plot(time, energies, 'b-', linewidth=1.0, label='Energy')
    ax1.axhline(y=threshold, color='red', linestyle='--', 
                linewidth=2.0, label='Threshold')
    ax1.axvline(x=true_drift, color='green', linestyle='-', 
                linewidth=2.5, label='True Drift')
    
    for i, det in enumerate(detections):
        label = 'Detected' if i == 0 else None
        ax1.axvline(x=det, color='orange', linestyle='--', 
                    linewidth=1.5, alpha=0.8, label=label)
    
    ax1.set_ylabel('Energy', fontsize=11)
    ax1.set_title(title, fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Detection timeline
    ax2 = axes[1]
    ax2.set_xlim(0, len(energies))
    ax2.set_ylim(0, 1)
    
    # Shade pre-drift as green, post-drift as red
    ax2.axvspan(0, true_drift, alpha=0.3, color='green', label='Normal')
    ax2.axvspan(true_drift, len(energies), alpha=0.3, color='red', label='Drift')
    
    # Mark detections
    for det in detections:
        ax2.scatter([det], [0.5], marker='v', s=100, color='orange', 
                    edgecolors='black', zorder=5)
    
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_yticks([])
    ax2.legend(loc='center right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = ['detection_delay', 'false_positive_rate', 'f1_score'],
    title: str = "Algorithm Comparison",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar plot comparing multiple algorithms on multiple metrics.
    
    Args:
        results: Dict mapping algorithm name to metrics dict
        metrics: Which metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Save path
        
    Returns:
        Figure
    """
    algorithms = list(results.keys())
    n_algorithms = len(algorithms)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_algorithms))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [results[alg].get(metric, 0) for alg in algorithms]
        
        bars = ax.bar(algorithms, values, color=colors)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_graph_energy(
    adjacency: np.ndarray,
    edge_energies: Dict[int, float],
    node_positions: Optional[np.ndarray] = None,
    title: str = "Sheaf Energy Distribution",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize sheaf energy on a graph.
    
    Edges are colored by their energy contribution.
    
    Args:
        adjacency: Graph adjacency matrix
        edge_energies: Dict mapping edge index to energy
        node_positions: Node positions for layout (optional)
        title: Plot title
        figsize: Figure size
        save_path: Save path
        
    Returns:
        Figure
    """
    n_nodes = adjacency.shape[0]
    
    # Generate positions if not provided
    if node_positions is None:
        # Simple circular layout
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        node_positions = np.column_stack([np.cos(angles), np.sin(angles)])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get edge list
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adjacency[i, j] > 0:
                edges.append((i, j))
    
    # Normalize energies for coloring
    if edge_energies:
        max_energy = max(edge_energies.values()) + 1e-6
        min_energy = min(edge_energies.values())
    else:
        max_energy, min_energy = 1, 0
    
    # Draw edges
    for idx, (i, j) in enumerate(edges):
        energy = edge_energies.get(idx, 0)
        normalized = (energy - min_energy) / (max_energy - min_energy)
        color = plt.cm.RdYlGn_r(normalized)  # Red = high energy
        
        ax.plot([node_positions[i, 0], node_positions[j, 0]],
                [node_positions[i, 1], node_positions[j, 1]],
                color=color, linewidth=2 + 3*normalized, alpha=0.7)
    
    # Draw nodes
    ax.scatter(node_positions[:, 0], node_positions[:, 1],
               s=200, c='lightblue', edgecolors='black', zorder=5)
    
    # Label nodes
    for i in range(n_nodes):
        ax.annotate(str(i), node_positions[i], ha='center', va='center',
                    fontsize=10, fontweight='bold')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                norm=plt.Normalize(vmin=min_energy, vmax=max_energy))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Edge Energy', fontsize=11)
    
    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
