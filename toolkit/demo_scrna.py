#!/usr/bin/env python3
"""
DEMO: TOPOLOGICAL ALIASING IN scRNA-seq
=======================================

THE QUESTION:
When computational biologists project 10,000+ dimensional gene expression
data into a 2D t-SNE or UMAP plot, how much of the apparent structure
is real vs. hallucinated?

THE APPROACH:
1. Load a standard scRNA-seq dataset
2. Compute D_sys (intrinsic dimensionality via participation ratio)
3. Compute 2D embedding (t-SNE)
4. Measure aliasing (how many 2D neighbors weren't high-D neighbors)
5. Visualize with the "Hairball of Truth"

THE RESULT:
Typically 60-70% aliasing. Two-thirds of the neighbors you see
in a t-SNE plot were NOT neighbors in the original high-D space.

THE IMPLICATION:
Every "cluster" you see in a 2D plot is partially hallucinated.
Every boundary you draw is wrong for ~60% of the cells near it.
This isn't a bug in the algorithm - it's geometry.

USAGE:
    python demo_scrna.py                    # Use synthetic data
    python demo_scrna.py --dataset pbmc3k   # Use PBMC 3k from scanpy

Author: Ian Todd
"""

import argparse
import numpy as np
import sys

# Import from the toolkit
from falsifiability import (
    generate_synthetic,
    load_scanpy_dataset,
    analyze_dataset,
    plot_metrics_dashboard,
    plot_hairball,
    compare_datasets,
    HAS_SCANPY
)


def run_single_dataset(dataset_name='synthetic'):
    """Analyze a single dataset."""

    print("=" * 60)
    print("TOPOLOGICAL ALIASING IN scRNA-seq")
    print("=" * 60)

    # Load data
    if dataset_name == 'synthetic':
        print("\nUsing synthetic data (no scanpy required)...")
        data, labels = generate_synthetic(n_samples=3000, n_dims=50)
        name = "Synthetic (50D, 5 clusters)"
        species = "N/A"
    else:
        if not HAS_SCANPY:
            print("ERROR: scanpy required for real datasets")
            print("Install with: pip install scanpy")
            print("Or run with: python demo_scrna.py (uses synthetic data)")
            sys.exit(1)

        print(f"\nLoading {dataset_name}...")
        data, name, species = load_scanpy_dataset(dataset_name)

    # Analyze
    results = analyze_dataset(data, name, verbose=True)
    results['species'] = species

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"""
    Dataset: {results['name']}
    Cells: {results['n_cells']:,}

    INTRINSIC DIMENSIONALITY:
      D_sys = {results['d_sys']:.1f}
      (The data effectively lives in ~{results['d_sys']:.0f} dimensions)

    PROJECTION:
      D_obs = 2 (t-SNE)
      Compression ratio = {results['d_sys']/2:.1f}x

    ALIASING:
      Topological: {results['aliasing']:.1%}
      → {results['aliasing']:.0%} of apparent neighbors in the 2D plot
         were NOT neighbors in the original space

      Cluster-based: {results['cluster_aliasing']:.1%}
      → A classifier trained on 2D coordinates gets
         {results['cluster_aliasing']:.0%} of cluster assignments WRONG

    COVERAGE:
      {results['coverage']:.2e} of the high-D space is sampled
      → The vast majority of possible states have NEVER been observed

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    WHAT THIS MEANS:

    When you look at a t-SNE plot and draw a boundary between
    "cell types," you are making a claim about which cells are
    similar. But {results['aliasing']:.0%} of the cells near that boundary
    are actually more similar to cells on the OTHER side.

    This isn't a failure of t-SNE. It's a geometric inevitability
    when you compress {results['d_sys']:.0f} dimensions into 2.

    The shadow lies. The question is what to do about it.
    """)

    # Plot
    print("Generating figures...")
    fig1 = plot_metrics_dashboard(results, save_path='fig_scrna_metrics.pdf')
    fig2 = plot_hairball(results['data_2d'], results['data_pca'],
                         n_lines=300, k=5, save_path='fig_scrna_hairball.pdf')

    import matplotlib.pyplot as plt
    plt.show()

    return results


def run_multi_dataset():
    """Compare aliasing across multiple datasets."""

    if not HAS_SCANPY:
        print("ERROR: scanpy required for multi-dataset comparison")
        sys.exit(1)

    print("=" * 60)
    print("MULTI-DATASET ALIASING COMPARISON")
    print("=" * 60)

    datasets = []
    for name in ['pbmc3k', 'paul15']:
        try:
            data, full_name, species = load_scanpy_dataset(name)
            datasets.append((data, full_name, species))
        except Exception as e:
            print(f"  Skipping {name}: {e}")

    if not datasets:
        print("No datasets loaded. Exiting.")
        sys.exit(1)

    results, summary = compare_datasets(datasets, verbose=True)

    print(f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    CONCLUSION:

    Across {summary['n_datasets']} datasets:
      - Average D_sys: {summary['avg_d_sys']:.1f}
      - Average aliasing: {summary['avg_aliasing']:.1%}

    This is NOT a quirk of one dataset or one method.
    This is geometry: when D_sys >> D_obs, topology is destroyed.

    The 2D plots that biologists use to define "cell types"
    are shadows that actively lie about similarity relationships.
    """)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure topological aliasing in scRNA-seq data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_scrna.py                    # Synthetic data (no dependencies)
    python demo_scrna.py --dataset pbmc3k   # PBMC 3k from 10X
    python demo_scrna.py --dataset paul15   # Paul et al. bone marrow
    python demo_scrna.py --multi            # Compare multiple datasets
        """
    )

    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'pbmc3k', 'paul15', 'pbmc68k'],
                        help='Dataset to analyze')
    parser.add_argument('--multi', action='store_true',
                        help='Compare multiple datasets')

    args = parser.parse_args()

    if args.multi:
        run_multi_dataset()
    else:
        run_single_dataset(args.dataset)


if __name__ == '__main__':
    main()
