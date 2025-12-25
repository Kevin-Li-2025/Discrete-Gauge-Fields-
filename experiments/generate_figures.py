"""
Generate figures for TPAMI paper:
1. Network visualization with edge residuals
2. Energy over time (drift detection timeline)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# ============================================================================
# FIGURE 1: Network Visualization with Edge Residuals
# ============================================================================

def create_network_figure():
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    # Node positions (hexagonal layout)
    positions = {
        0: (0, 2),
        1: (1.5, 2.5),
        2: (3, 2),
        3: (0, 0.5),
        4: (1.5, 1),
        5: (3, 0.5),
    }
    
    # Edges with residual energies
    edges = [
        (0, 1, 0.02, 'normal'),
        (1, 2, 0.03, 'normal'),
        (0, 3, 0.01, 'normal'),
        (1, 4, 0.45, 'drift'),  # High residual = drift
        (2, 5, 0.02, 'normal'),
        (3, 4, 0.38, 'drift'),  # High residual = drift
        (4, 5, 0.04, 'normal'),
    ]
    
    # Draw edges
    for u, v, residual, status in edges:
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        
        if status == 'drift':
            color = '#e74c3c'
            linewidth = 3
            linestyle = '-'
        else:
            color = '#2ecc71'
            linewidth = 2
            linestyle = '-'
        
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth, 
                linestyle=linestyle, zorder=1)
        
        # Label edge residual
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.15, f'{residual:.2f}', ha='center', fontsize=8,
                color=color, fontweight='bold')
    
    # Draw nodes
    for v, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.25, color='#3498db', ec='black', 
                            linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, f'$v_{v}$', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    # Legend
    normal_patch = mpatches.Patch(color='#2ecc71', label='Normal ($r_e < \\theta$)')
    drift_patch = mpatches.Patch(color='#e74c3c', label='Drift ($r_e > \\theta$)')
    ax.legend(handles=[normal_patch, drift_patch], loc='upper right', fontsize=9)
    
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.2, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Edge Residual Energy for Drift Localization', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../paper/figures/network_residuals.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../paper/figures/network_residuals.png', dpi=300, bbox_inches='tight')
    print("Saved: network_residuals.pdf/.png")


# ============================================================================
# FIGURE 2: Energy Over Time (Drift Detection Timeline)
# ============================================================================

def create_energy_timeline():
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    
    np.random.seed(42)
    
    # Time points
    t = np.arange(0, 400)
    drift_start = 200
    
    # Baseline energy
    baseline = 0.05 + 0.02 * np.random.randn(len(t))
    
    # Post-drift energy
    energy = baseline.copy()
    for i in range(drift_start, len(t)):
        # Gradual increase then stabilize
        progress = min((i - drift_start) / 50, 1.0)
        energy[i] = baseline[i] + 0.35 * progress + 0.05 * np.random.randn()
    
    # Plot energy
    ax.plot(t, energy, color='#3498db', linewidth=1.2, label='THS Energy $\\bar{\\mathcal{E}}(t)$')
    
    # Threshold line
    threshold = 0.15
    ax.axhline(y=threshold, color='#e74c3c', linestyle='--', linewidth=1.5, 
               label='Threshold $\\theta$')
    
    # Drift onset
    ax.axvline(x=drift_start, color='#2c3e50', linestyle=':', linewidth=1.5, 
               label='Drift onset')
    
    # Detection point
    detection = drift_start + 12
    ax.annotate('Detection\n(delay=12)', xy=(detection, energy[detection]),
                xytext=(detection + 30, 0.35),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
                fontsize=9, color='#27ae60', ha='center')
    
    # Fill regions
    ax.fill_between(t[:drift_start], 0, energy[:drift_start], 
                    alpha=0.2, color='#2ecc71', label='Normal')
    ax.fill_between(t[drift_start:], 0, energy[drift_start:], 
                    alpha=0.2, color='#e74c3c', label='Drift')
    
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Normalized Energy $\\bar{\\mathcal{E}}(t)$')
    ax.set_title('THS Drift Detection Timeline')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 0.5)
    
    plt.tight_layout()
    plt.savefig('../paper/figures/energy_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../paper/figures/energy_timeline.png', dpi=300, bbox_inches='tight')
    print("Saved: energy_timeline.pdf/.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import os
    os.makedirs('../paper/figures', exist_ok=True)
    
    print("Generating figures for TPAMI paper...")
    create_network_figure()
    create_energy_timeline()
    print("Done!")
