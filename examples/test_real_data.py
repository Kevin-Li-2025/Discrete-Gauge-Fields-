#!/usr/bin/env python3
"""
Test THS-Drift on real-world datasets.

Uses actual data from scikit-learn and simulates
drift by mixing classes or temporal splits.
"""

import numpy as np
from sklearn.datasets import load_digits, load_wine, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from ths import THSDrift
from ths.drift.baselines import PCADrift


def test_digits_dataset():
    """
    Test on MNIST-like digits dataset.
    
    Simulate drift by training on digits 0-4,
    then testing on digits 5-9.
    """
    print("=" * 60)
    print("TEST 1: Handwritten Digits (Class Shift)")
    print("=" * 60)
    
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split: train on digits 0-4, test includes both
    train_mask = y <= 4
    X_train = X[train_mask][:200]
    
    # Test set: first 100 from 0-4 (no drift), then 100 from 5-9 (drift)
    X_test_normal = X[train_mask][200:300]
    X_test_drift = X[~train_mask][:100]
    X_test = np.vstack([X_test_normal, X_test_drift])
    
    print(f"Training: 200 samples (digits 0-4)")
    print(f"Test: 100 normal + 100 drifted (digits 5-9)")
    print(f"Expected drift at sample 100")
    
    # THS-Drift
    detector = THSDrift(dim=5000, k=10, alpha=2.0, persistence=5, seed=42)
    detector.fit(X_train)
    
    energies = []
    first_detection = None
    
    for i, x in enumerate(X_test):
        energy = detector.update(x)
        energies.append(energy)
        if detector.detect().is_drift and first_detection is None:
            first_detection = i
    
    print(f"\nResults:")
    print(f"  Pre-drift mean energy: {np.mean(energies[:100]):.4f}")
    print(f"  Post-drift mean energy: {np.mean(energies[100:]):.4f}")
    print(f"  First detection at: {first_detection}")
    if first_detection and first_detection >= 100:
        print(f"  Detection delay: {first_detection - 100} samples")
    
    return energies


def test_wine_dataset():
    """
    Test on Wine dataset.
    
    Simulate drift by training on class 0,
    then introducing class 1 and 2.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Wine Classification (Distribution Shift)")
    print("=" * 60)
    
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train on class 0 only
    train_mask = y == 0
    X_train = X[train_mask]
    
    # Test: class 0, then class 1, then class 2
    X_test_c0 = X[y == 0][:20]
    X_test_c1 = X[y == 1][:30]
    X_test_c2 = X[y == 2][:30]
    X_test = np.vstack([X_test_c0, X_test_c1, X_test_c2])
    
    print(f"Training: {len(X_train)} samples (class 0 only)")
    print(f"Test: 20 class0 + 30 class1 + 30 class2")
    print(f"Expected drift at sample 20")
    
    detector = THSDrift(dim=5000, k=5, alpha=2.5, persistence=3, seed=42)
    detector.fit(X_train)
    
    energies = []
    first_detection = None
    
    for i, x in enumerate(X_test):
        energy = detector.update(x)
        energies.append(energy)
        if detector.detect().is_drift and first_detection is None:
            first_detection = i
    
    print(f"\nResults:")
    print(f"  Class 0 mean energy: {np.mean(energies[:20]):.4f}")
    print(f"  Class 1 mean energy: {np.mean(energies[20:50]):.4f}")
    print(f"  Class 2 mean energy: {np.mean(energies[50:]):.4f}")
    print(f"  First detection at: {first_detection}")
    
    return energies


def test_housing_temporal():
    """
    Test on California Housing with temporal split.
    
    Simulate drift by using first half as reference,
    and detecting change in second half.
    """
    print("\n" + "=" * 60)
    print("TEST 3: California Housing (Temporal Simulation)")
    print("=" * 60)
    
    housing = fetch_california_housing()
    X = housing.data
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    n = len(X)
    print(f"Dataset: {n} samples, {X.shape[1]} features")
    
    # Shuffle and create artificial drift by scaling
    np.random.seed(42)
    indices = np.random.permutation(n)
    X = X[indices]
    
    # Take subset for speed
    X = X[:2000]
    
    # Train on first 500
    X_train = X[:500]
    
    # Test: 500 normal, then 500 with artificial drift (scaled features)
    X_test_normal = X[500:1000]
    X_test_drift = X[1000:1500] * 1.5 + 0.5  # Scale and shift
    X_test = np.vstack([X_test_normal, X_test_drift])
    
    print(f"Training: 500 samples")
    print(f"Test: 500 normal + 500 artificially shifted")
    print(f"Expected drift at sample 500")
    
    detector = THSDrift(dim=5000, k=10, alpha=2.5, persistence=5, seed=42)
    detector.fit(X_train)
    
    energies = []
    first_detection = None
    
    for i, x in enumerate(X_test):
        energy = detector.update(x)
        energies.append(energy)
        if detector.detect().is_drift and first_detection is None:
            first_detection = i
    
    print(f"\nResults:")
    print(f"  Pre-drift mean energy: {np.mean(energies[:500]):.4f}")
    print(f"  Post-drift mean energy: {np.mean(energies[500:]):.4f}")
    print(f"  First detection at: {first_detection}")
    if first_detection and first_detection >= 500:
        print(f"  Detection delay: {first_detection - 500} samples")
    
    return energies


def compare_with_pca():
    """Compare THS vs PCA on real data."""
    print("\n" + "=" * 60)
    print("COMPARISON: THS-Drift vs PCA-CD on Digits")
    print("=" * 60)
    
    digits = load_digits()
    X, y = digits.data, digits.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    train_mask = y <= 4
    X_train = X[train_mask][:200]
    X_test_normal = X[train_mask][200:300]
    X_test_drift = X[~train_mask][:100]
    X_test = np.vstack([X_test_normal, X_test_drift])
    
    # THS
    ths = THSDrift(dim=5000, k=10, alpha=2.0, persistence=5, seed=42)
    ths.fit(X_train)
    
    ths_detection = None
    for i, x in enumerate(X_test):
        ths.update(x)
        if ths.detect().is_drift and ths_detection is None:
            ths_detection = i
    
    # PCA
    pca = PCADrift(n_components=20, alpha=2.0)
    pca.fit(X_train)
    
    pca_detection = None
    for i, x in enumerate(X_test):
        pca.update(x)
        if pca.detect().is_drift and pca_detection is None:
            pca_detection = i
    
    print(f"\nDrift expected at sample 100")
    print(f"THS-Drift detection: {ths_detection}")
    print(f"PCA-CD detection: {pca_detection}")
    
    if ths_detection and ths_detection >= 100:
        print(f"THS delay: {ths_detection - 100}")
    if pca_detection and pca_detection >= 100:
        print(f"PCA delay: {pca_detection - 100}")


def main():
    print("\n" + "#" * 60)
    print("# THS-Drift Testing on Real-World Datasets")
    print("#" * 60)
    
    test_digits_dataset()
    test_wine_dataset()
    test_housing_temporal()
    compare_with_pca()
    
    print("\n" + "=" * 60)
    print("All real-world tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
