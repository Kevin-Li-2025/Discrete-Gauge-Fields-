#!/usr/bin/env python3
"""
Energy Benchmark: THS vs Autoencoder vs PCA.

Compares computational efficiency (operations, estimated energy)
of different drift detection methods.
"""

import numpy as np
import time
from typing import Dict, List
from ths import THSDrift
from ths.drift.baselines import PCADrift
from ths.utils.metrics import EnergyProfiler

# Try to import torch for autoencoder
try:
    from ths.drift.baselines import AutoencoderDrift
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def benchmark_throughput(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_warmup: int = 10
) -> Dict[str, Dict]:
    """
    Benchmark inference throughput for different methods.
    
    Args:
        X_train: Training data
        X_test: Test data for inference
        n_warmup: Warmup iterations
        
    Returns:
        Results for each method
    """
    results = {}
    n_features = X_train.shape[1]
    n_test = len(X_test)
    
    # THS-Drift
    print("\nBenchmarking THS-Drift...")
    ths = THSDrift(dim=10000, k=5, seed=42)
    ths.fit(X_train)
    
    # Warmup
    for i in range(min(n_warmup, n_test)):
        ths.update(X_test[i])
    ths.reset()
    
    # Time
    start = time.perf_counter()
    for x in X_test:
        ths.update(x)
    elapsed = time.perf_counter() - start
    
    ths_throughput = n_test / elapsed
    ths_energy = EnergyProfiler.estimate_ths_energy(
        dim=10000, 
        n_edges=ths.sheaf.graph.n_edges()
    )
    
    results['THS-Drift'] = {
        'throughput': ths_throughput,
        'latency_ms': 1000 / ths_throughput,
        'energy_per_sample_uj': ths_energy,
        'n_edges': ths.sheaf.graph.n_edges(),
    }
    print(f"  Throughput: {ths_throughput:.1f} samples/sec")
    
    # PCA-CD
    print("\nBenchmarking PCA-CD...")
    pca = PCADrift(n_components=10)
    pca.fit(X_train)
    
    start = time.perf_counter()
    for x in X_test:
        pca.update(x)
    elapsed = time.perf_counter() - start
    
    pca_throughput = n_test / elapsed
    # PCA operations: project (n_features * n_components) + reconstruct + norm
    pca_ops = n_features * 10 * 2 + n_features
    pca_energy = pca_ops * (3.7 + 7.5) / 1e6  # Approximate uJ
    
    results['PCA-CD'] = {
        'throughput': pca_throughput,
        'latency_ms': 1000 / pca_throughput,
        'energy_per_sample_uj': pca_energy,
    }
    print(f"  Throughput: {pca_throughput:.1f} samples/sec")
    
    # Autoencoder (if available)
    if TORCH_AVAILABLE:
        print("\nBenchmarking Autoencoder...")
        hidden_dims = [64, 32, 64]
        ae = AutoencoderDrift(hidden_dims=hidden_dims, epochs=50, device='cpu')
        ae.fit(X_train)
        
        start = time.perf_counter()
        for x in X_test:
            ae.update(x)
        elapsed = time.perf_counter() - start
        
        ae_throughput = n_test / elapsed
        ae_energy = EnergyProfiler.estimate_autoencoder_energy(
            n_features, hidden_dims
        )
        
        results['Autoencoder'] = {
            'throughput': ae_throughput,
            'latency_ms': 1000 / ae_throughput,
            'energy_per_sample_uj': ae_energy,
        }
        print(f"  Throughput: {ae_throughput:.1f} samples/sec")
    
    return results


def print_comparison_table(results: Dict[str, Dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"\n{'Method':<15} {'Throughput':>15} {'Latency':>12} {'Energy':>15}")
    print(f"{'':15} {'(samples/s)':>15} {'(ms)':>12} {'(µJ/sample)':>15}")
    print("-" * 70)
    
    for name, metrics in results.items():
        print(f"{name:<15} "
              f"{metrics['throughput']:>15.1f} "
              f"{metrics['latency_ms']:>12.3f} "
              f"{metrics['energy_per_sample_uj']:>15.2f}")
    
    print("-" * 70)
    
    # Compute ratios relative to THS
    if 'THS-Drift' in results:
        ths = results['THS-Drift']
        print("\nRelative to THS-Drift:")
        for name, metrics in results.items():
            if name != 'THS-Drift':
                speed_ratio = ths['throughput'] / metrics['throughput']
                energy_ratio = metrics['energy_per_sample_uj'] / ths['energy_per_sample_uj']
                print(f"  {name}: {speed_ratio:.1f}x slower, {energy_ratio:.1f}x more energy")


def analyze_dimensionality_scaling():
    """Analyze how THS scales with hypervector dimension."""
    print("\n" + "=" * 70)
    print("DIMENSIONALITY SCALING ANALYSIS")
    print("=" * 70)
    
    dims = [1000, 2000, 5000, 10000, 20000]
    n_samples = 100
    n_features = 10
    
    X_train = np.random.randn(200, n_features)
    X_test = np.random.randn(n_samples, n_features)
    
    print(f"\n{'Dimension':>10} {'Throughput':>15} {'Energy (µJ)':>15}")
    print("-" * 45)
    
    for dim in dims:
        ths = THSDrift(dim=dim, k=5, seed=42)
        ths.fit(X_train)
        
        start = time.perf_counter()
        for x in X_test:
            ths.update(x)
        elapsed = time.perf_counter() - start
        
        throughput = n_samples / elapsed
        energy = EnergyProfiler.estimate_ths_energy(
            dim=dim,
            n_edges=ths.sheaf.graph.n_edges()
        )
        
        print(f"{dim:>10} {throughput:>15.1f} {energy:>15.3f}")


def estimate_edge_device_performance():
    """Estimate performance on typical edge devices."""
    print("\n" + "=" * 70)
    print("EDGE DEVICE PERFORMANCE ESTIMATES")
    print("=" * 70)
    
    # Device profiles (relative to laptop CPU)
    devices = {
        'Laptop (Intel i7)': {'clock_factor': 1.0, 'power_w': 45},
        'Raspberry Pi 4': {'clock_factor': 0.15, 'power_w': 5},
        'ARM Cortex-M4': {'clock_factor': 0.02, 'power_w': 0.1},
        'ESP32': {'clock_factor': 0.01, 'power_w': 0.05},
    }
    
    # Baseline: THS on laptop
    dim = 10000
    n_edges = 50  # Typical for k=5, 200 samples
    base_energy_uj = EnergyProfiler.estimate_ths_energy(dim, n_edges)
    base_throughput = 50000  # Approximate from benchmarks
    
    print(f"\nTHS-Drift (dim={dim}, ~{n_edges} edges)")
    print(f"\n{'Device':<25} {'Throughput':>12} {'Energy/Sample':>15} {'Battery Life*':>15}")
    print("-" * 70)
    
    for device, profile in devices.items():
        throughput = base_throughput * profile['clock_factor']
        # Scale energy by power ratio
        energy_uj = base_energy_uj * (profile['power_w'] / 45)
        
        # Battery life estimate (2000 mAh, 3.7V coin cell = 26.6 kJ)
        battery_capacity_uj = 26.6e9  # 2000 mAh * 3.7V in uJ
        samples_on_battery = battery_capacity_uj / energy_uj
        hours = samples_on_battery / (throughput * 3600)
        
        print(f"{device:<25} {throughput:>12.0f}/s {energy_uj:>15.2f} µJ {hours:>12.0f} hours")
    
    print("\n* Assuming continuous inference on 2000mAh 3.7V battery")


def main():
    """Run all benchmarks."""
    print("\n" + "#" * 70)
    print("# THS Energy Efficiency Benchmark")
    print("#" * 70)
    
    # Generate data
    np.random.seed(42)
    n_features = 20
    X_train = np.random.randn(300, n_features)
    X_test = np.random.randn(500, n_features)
    
    print(f"\nData: {len(X_train)} train, {len(X_test)} test, {n_features} features")
    
    # Run benchmarks
    results = benchmark_throughput(X_train, X_test)
    print_comparison_table(results)
    
    analyze_dimensionality_scaling()
    estimate_edge_device_performance()
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
