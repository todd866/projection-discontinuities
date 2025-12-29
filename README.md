# The Limits of Falsifiability

**Published in BioSystems 258, 105608 (2025)**

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.biosystems.2025.105608-blue)](https://doi.org/10.1016/j.biosystems.2025.105608)

## Overview

When biological systems operate in high-dimensional state spaces but we observe only low-dimensional projections, classical falsifiability becomes geometrically incoherent. This paper develops the theory of how dimensional projection destroys the epistemic conditions required for binary hypothesis testing.

**Key result:** When D_sys >> D_obs, topological aliasing makes it impossible to distinguish between structurally distinct hypotheses from projected observations alone.

## Version History

This paper follows a **living document** approach, with periodic upgrades that extend the theoretical framework while preserving backward compatibility with the published version.

### v1.0 (October 2025)
- Published in BioSystems
- Core arguments: dimensional projection, sub-Landauer domain, stochastic resonance
- DOI: [10.1016/j.biosystems.2025.105608](https://doi.org/10.1016/j.biosystems.2025.105608)

### v2.0 (December 2025)
- **New section: Framework Dependence** - The deepest limitation
- Extended Duhem-Quine thesis: framework choice is itself a projection
- Wigner's "unreasonable effectiveness" reframed as selection bias
- Three levels of limitation: physical, dimensional, and axiomatic
- New figures visualizing framework projection and selection bias
- Available in: `v2.0/` folder

**Why version papers?** AI-assisted research tools enable rapid theoretical development. Rather than waiting for formal publication cycles, we release upgraded versions that extend and refine the arguments. Each version is self-contained and citable. The published version remains the canonical reference for formal citation.

## Repository Structure

```
1_falsifiability/
├── figures/                    # v1.0 paper figures
├── submission_package/         # Final v1.0 submission materials
├── archive/                    # Archived companion materials
├── v2.0/                       # Version 2.0 (December 2025)
│   ├── falsifiability_v2.tex   # Upgraded manuscript
│   ├── falsifiability_v2.pdf   # Compiled PDF
│   └── figures/                # New figures + generation scripts
└── README.md
```

## Key Arguments

### v1.0 (Published)
1. **Dimensional Projection Loss**: Binary tests on high-D systems preserve ~0% of information
2. **Sub-Landauer Domain**: Many biological patterns exist below measurement thresholds
3. **Stochastic Resonance**: Weak signals detectable only through population-level pooling

### v2.0 (Extended)
4. **Framework Dependence**: Before any measurement, axiomatic choices have already made a dimensional reduction
5. **Mathematics as Projection**: All math is finite-dimensional; every equation is a shadow of high-D reality
6. **Wigner Selection Bias**: Physics "works" because we study domains where projection loss is small

## Citation

For the published version:
```bibtex
@article{todd2025limits,
  title={The limits of falsifiability},
  author={Todd, Ian},
  journal={BioSystems},
  volume={258},
  pages={105608},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.biosystems.2025.105608}
}
```

For the extended version (v2.0):
```bibtex
@misc{todd2025limitsv2,
  title={The Limits of Falsifiability: Dimensionality, Measurement Thresholds, and the Sub-Landauer Domain in Biological Systems (Version 2.0)},
  author={Todd, Ian},
  year={2025},
  note={Extended version available at: https://github.com/todd866/limits-of-falsifiability}
}
```

## Author

Ian Todd
Sydney Medical School, University of Sydney
itod2305@uni.sydney.edu.au
ORCID: [0009-0002-6994-0917](https://orcid.org/0009-0002-6994-0917)

## License

MIT
