#!/usr/bin/env python3
"""
Concept Drift Detection Demo.

Demonstrates THS-Drift detecting concept drift on synthetic data
and compares with baseline methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from ths import THSDrift
from ths.datasets.synthetic import SuddenDrift, RotatingGaussian
from ths.drift.baselines import PCADrift, ADWINDrift
from ths.utils.metrics import evaluate_detector
from ths.utils.visualization import plot_drift_detection


def demo_sudden_drift():
    """Detect sudden distribution change."""
    print("=" * 60)
    print("SUDDEN DRIFT DETECTION")
    print("=" * 60)
    
    # Generate data with sudden drift at t=500
    gen = SuddenDrift(
        n_samples=1000,
        drift_point=500,
        dim=10,
        seed=42
    )
    
    samples = list(gen)
    data = np.array([s.data for s in samples])
    
    # Split into train/test
    X_train = data[:300]
    X_test = data[300:]
    drift_point = 500 - 300  # Adjust for test set offset
    
    print(f"\nData: {len(data)} samples, 10 features")
    print(f"Training: first 300 samples")
    print(f"Drift occurs at sample 500 (sample 200 in test set)")
    
    # Initialize THS-Drift
    detector = THSDrift(
        dim=5000,
        k=5,
        alpha=2.5,
        persistence=3,
        seed=42
    )
    
    # Fit on reference data
    print("\nFitting THS-Drift on reference data...")
    detector.fit(X_train)
    
    # Monitor test data
    print("Monitoring test data for drift...")
    energies = []
    detections = []
    
    for i, x in enumerate(X_test):
        energy = detector.update(x)
        energies.append(energy)
        
        result = detector.detect()
        if result.is_drift and i not in [d for d in detections]:
            detections.append(i)
            print(f"  Drift detected at test sample {i} (actual: {i + 300})")
            print(f"    Energy: {result.energy:.4f}, Threshold: {result.threshold:.4f}")
            break  # Stop after first detection
    
    # Continue to collect energy for plotting
    for i in range(len(energies), len(X_test)):
        x = X_test[i]
        energy = detector.update(x)
        energies.append(energy)
    
    # Evaluate
    true_drift = [drift_point]
    metrics = evaluate_detector(true_drift, detections, len(X_test))
    
    print(f"\nResults:")
    print(f"  Detection delay: {metrics['detection_delay']:.1f} samples")
    print(f"  True positive rate: {metrics['true_positive_rate']:.2f}")
    print(f"  False positive rate: {metrics['false_positive_rate']:.4f}")
    
    # Get threshold
    stats = detector.get_statistics()
    threshold = stats['threshold']
    
    # Plot
    fig = plot_drift_detection(
        energies=np.array(energies),
        detections=detections,
        true_drift=drift_point,
        threshold=threshold,
        title="THS-Drift: Sudden Drift Detection"
    )
    plt.savefig("sudden_drift_demo.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to: sudden_drift_demo.png")
    plt.close()
    
    return detector, energies, detections


def demo_rotating_gaussian():
    """Detect gradual drift in rotating Gaussian."""
    print("\n" + "=" * 60)
    print("ROTATING GAUSSIAN (GRADUAL DRIFT)")
    print("=" * 60)
    
    gen = RotatingGaussian(
        n_samples=800,
        drift_start=400,
        rotation_speed=0.03,
        seed=42
    )
    
    data, concepts, drift_point = gen.generate_array()
    
    X_train = data[:300]
    X_test = data[300:]
    adjusted_drift = drift_point - 300
    
    print(f"\nData: 2D rotating Gaussian cloud")
    print(f"Rotation begins at sample 400 (sample 100 in test)")
    
    detector = THSDrift(
        dim=5000,
        k=5,
        alpha=2.0,
        persistence=5,
        seed=42
    )
    
    detector.fit(X_train)
    
    energies = []
    detections = []
    
    for i, x in enumerate(X_test):
        energy = detector.update(x)
        energies.append(energy)
        
        result = detector.detect()
        if result.is_drift and not detections:
            detections.append(i)
            print(f"  Drift detected at test sample {i}")
    
    print(f"\nRotating Gaussian test complete.")
    print(f"  First detection at: {detections[0] if detections else 'None'}")
    print(f"  Expected drift region starts at: {adjusted_drift}")


def compare_detectors():
    """Compare THS-Drift with baseline methods."""
    print("\n" + "=" * 60)
    print("DETECTOR COMPARISON")
    print("=" * 60)
    
    # Generate data
    gen = SuddenDrift(n_samples=800, drift_point=400, dim=10, seed=42)
    samples = list(gen)
    data = np.array([s.data for s in samples])
    
    X_train = data[:200]
    X_test = data[200:]
    true_drift = [200]  # Relative to test set
    
    results = {}
    
    # THS-Drift
    print("\n1. THS-Drift")
    ths = THSDrift(dim=5000, k=5, alpha=2.5, persistence=3, seed=42)
    ths.fit(X_train)
    
    ths_detections = []
    for i, x in enumerate(X_test):
        ths.update(x)
        if ths.detect().is_drift and not ths_detections:
            ths_detections.append(i)
            print(f"   Detected at sample {i}")
            break
    
    results['THS-Drift'] = evaluate_detector(true_drift, ths_detections, len(X_test))
    
    # PCA-CD
    print("\n2. PCA-CD")
    pca = PCADrift(n_components=5, alpha=2.5)
    pca.fit(X_train)
    
    pca_detections = []
    for i, x in enumerate(X_test):
        pca.update(x)
        if pca.detect().is_drift and not pca_detections:
            pca_detections.append(i)
            print(f"   Detected at sample {i}")
            break
    
    results['PCA-CD'] = evaluate_detector(true_drift, pca_detections, len(X_test))
    
    # ADWIN
    print("\n3. ADWIN")
    adwin = ADWINDrift(delta=0.002)
    adwin.fit(X_train)
    
    adwin_detections = []
    for i, x in enumerate(X_test):
        adwin.update(x)
        if adwin.detect().is_drift and not adwin_detections:
            adwin_detections.append(i)
            print(f"   Detected at sample {i}")
            break
    
    results['ADWIN'] = evaluate_detector(true_drift, adwin_detections, len(X_test))
    
    # Summary table
    print("\n" + "-" * 50)
    print("COMPARISON SUMMARY")
    print("-" * 50)
    print(f"{'Method':<15} {'Delay':>10} {'TPR':>10} {'FPR':>10}")
    print("-" * 50)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['detection_delay']:>10.1f} "
              f"{metrics['true_positive_rate']:>10.2f} "
              f"{metrics['false_positive_rate']:>10.4f}")


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# THS Concept Drift Detection Demo")
    print("#" * 60)
    
    demo_sudden_drift()
    demo_rotating_gaussian()
    compare_detectors()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
