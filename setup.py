"""
Setup script for the projection-discontinuities toolkit.

Install with:
    pip install .

Or for development:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="projection-discontinuities",
    version="1.0.0",
    author="Ian Todd",
    author_email="itod2305@uni.sydney.edu.au",
    description="Quantify topological aliasing and projection-induced discontinuities in high-dimensional data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/todd866/projection-discontinuities",
    py_modules=["projection_discontinuities"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "scanpy": ["scanpy>=1.9.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="topological aliasing, dimensionality reduction, t-SNE, UMAP, Lorenz attractor, nonlinear dynamics",
)
