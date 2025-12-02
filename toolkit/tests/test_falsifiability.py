"""
Unit tests for the falsifiability toolkit.

Run with: pytest toolkit/tests/
"""
import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falsifiability import (
    generate_synthetic,
    participation_ratio,
    compute_aliasing,
    compute_coverage,
    analyze_dataset,
)


class TestGenerateSynthetic:
    """Tests for synthetic data generation."""

    def test_returns_correct_shape(self):
        """Generated data should have requested dimensions."""
        data, labels = generate_synthetic(n_samples=100, n_dims=20, n_clusters=3)
        assert data.shape == (100, 20)
        assert labels.shape == (100,)

    def test_correct_number_of_clusters(self):
        """Labels should contain expected number of unique clusters."""
        data, labels = generate_synthetic(n_samples=300, n_dims=10, n_clusters=5)
        assert len(np.unique(labels)) == 5

    def test_data_is_numeric(self):
        """Generated data should be numeric floats."""
        data, labels = generate_synthetic(n_samples=50, n_dims=10)
        assert np.issubdtype(data.dtype, np.floating)
        assert not np.any(np.isnan(data))


class TestParticipationRatio:
    """Tests for intrinsic dimensionality estimation."""

    def test_returns_float(self):
        """D_sys should be a float."""
        data = np.random.randn(100, 20)
        d_sys, eigenvalues = participation_ratio(data)
        assert isinstance(d_sys, (float, np.floating))

    def test_reasonable_range(self):
        """D_sys should be between 1 and n_dims."""
        data = np.random.randn(100, 20)
        d_sys, eigenvalues = participation_ratio(data)
        assert 1 <= d_sys <= 20

    def test_low_dimensional_data(self):
        """Data confined to subspace should have low D_sys."""
        # Create data that lives in a 3D subspace of 20D
        np.random.seed(42)
        basis = np.random.randn(3, 20)
        coeffs = np.random.randn(200, 3)
        data = coeffs @ basis

        d_sys, _ = participation_ratio(data)
        # Should be close to 3 (the true dimensionality)
        assert d_sys < 5

    def test_returns_eigenvalues(self):
        """Should return eigenvalues array."""
        data = np.random.randn(100, 20)
        d_sys, eigenvalues = participation_ratio(data)
        assert isinstance(eigenvalues, np.ndarray)
        assert len(eigenvalues) > 0


class TestComputeAliasing:
    """Tests for topological aliasing computation."""

    def test_returns_tuple(self):
        """Should return (aliasing, per_point_scores) tuple."""
        data_high = np.random.randn(100, 20)
        data_low = np.random.randn(100, 2)

        result = compute_aliasing(data_high, data_low, k=5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_aliasing_in_valid_range(self):
        """Aliasing rate should be in [0, 1]."""
        data_high = np.random.randn(100, 20)
        data_low = np.random.randn(100, 2)

        aliasing, _ = compute_aliasing(data_high, data_low, k=5)
        assert 0 <= aliasing <= 1

    def test_perfect_projection_low_aliasing(self):
        """PCA projection should have lower aliasing than random."""
        from sklearn.decomposition import PCA

        np.random.seed(42)
        data_high = np.random.randn(200, 10)

        # PCA projection (preserves structure)
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_high)

        # Random projection (destroys structure)
        data_random = np.random.randn(200, 2)

        aliasing_pca, _ = compute_aliasing(data_high, data_pca, k=5)
        aliasing_random, _ = compute_aliasing(data_high, data_random, k=5)

        # PCA should have less aliasing than random
        assert aliasing_pca < aliasing_random


class TestComputeCoverage:
    """Tests for state space coverage computation."""

    def test_returns_tuple(self):
        """Should return (coverage, occupied, total) tuple."""
        data = np.random.randn(100, 10)
        result = compute_coverage(data, n_bins=3, n_dims=5)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_coverage_in_valid_range(self):
        """Coverage should be in [0, 1]."""
        data = np.random.randn(100, 10)
        coverage, occupied, total = compute_coverage(data, n_bins=3, n_dims=5)

        assert 0 <= coverage <= 1
        assert occupied <= total

    def test_more_samples_higher_coverage(self):
        """More samples should generally give higher coverage."""
        np.random.seed(42)
        data_small = np.random.randn(50, 10)
        data_large = np.random.randn(500, 10)

        coverage_small, _, _ = compute_coverage(data_small, n_bins=3, n_dims=5)
        coverage_large, _, _ = compute_coverage(data_large, n_bins=3, n_dims=5)

        assert coverage_large >= coverage_small


class TestAnalyzeDataset:
    """Integration tests for the main analysis function."""

    def test_returns_dict(self):
        """Results should be a dictionary."""
        data, _ = generate_synthetic(n_samples=200, n_dims=20)
        results = analyze_dataset(data, name="Test", verbose=False)

        assert isinstance(results, dict)

    def test_contains_core_metrics(self):
        """Results dict should contain core metrics."""
        data, _ = generate_synthetic(n_samples=200, n_dims=20)
        results = analyze_dataset(data, name="Test", verbose=False)

        # These are the metrics we care about
        assert 'd_sys' in results
        assert 'aliasing' in results
        assert 'name' in results

    def test_synthetic_data_has_aliasing(self):
        """High-D to 2D projection should show aliasing."""
        data, _ = generate_synthetic(n_samples=500, n_dims=50)
        results = analyze_dataset(data, name="Test", verbose=False)

        # Aliasing should be present and positive
        assert results['aliasing'] > 0


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_2d_input_data(self):
        """Should handle already-2D data."""
        data = np.random.randn(100, 2)
        d_sys, _ = participation_ratio(data)
        assert d_sys <= 2

    def test_high_k_neighbors(self):
        """Should handle k approaching n_samples."""
        data_high = np.random.randn(50, 10)
        data_low = np.random.randn(50, 2)

        # k=10 is reasonable for 50 samples
        aliasing, _ = compute_aliasing(data_high, data_low, k=10)
        assert 0 <= aliasing <= 1
