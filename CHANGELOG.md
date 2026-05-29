# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-05-29

### Added

- **`FieldStrengthSplit`** — electrostatic field-strength greedy selection; deterministic O(N²) split that maximises coverage of the feature space.
- **`SpectralSplit`** — RBF affinity → normalised Laplacian → spectral embedding → k-means cluster assignment; exposes the cluster structure of the data across train/test.
- **`CombinatorialPurgedKFold`** — combinatorial purged cross-validation (CPCV, de Prado 2018); generates all C(k, n\_test) purged walk-forward pairs for a more exhaustive evaluation on financial time series.
- **`OnionSplit`** — peels concentric shells (Onion method, Gallagher & O'Sullivan 2022) to produce a training set that covers the boundary of the feature space.
- **`XYOnionSplit`** — joint X–y variant of `OnionSplit`; distance from the centroid is measured in normalised SPXY space so outer layers cover both feature and response diversity.
- **`DuplexSplit`** / **`LazyDuplexSplit`** — simultaneous maximin selection for train *and* test (Duplex algorithm, Snee 1977); unlike Kennard–Stone, both cohorts independently cover the feature space.
- **`VenetianBlindsCV`** — round-robin fold assignment over samples sorted by target value; guarantees a perfectly uniform spread of the target range across all folds without binning (Kennedy & Halvorsen 1992).
- `LinearAlgebra` added to explicit package dependencies (required by Julia 1.12+).
- CI now validates against Julia 1.9 (the declared minimum) in addition to the latest 1.x release.

## [0.2.0] - 2025-05-28

Initial public release.
