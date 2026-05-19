```@meta
CurrentModule = DataSplits
```

# Kennard–Stone

The Kennard–Stone algorithm (Kennard & Stone, 1969) — also known as CADEX — selects
a training set that **covers the feature space as uniformly as possible**. It is the
most widely used rational splitting method for spectroscopic and tabular calibration
data.

## How it works

1. Compute all pairwise distances between samples.
2. Select the two samples with the largest mutual distance as the first two training
   points.
3. Iteratively select the sample that has the **largest minimum distance** to any
   already-selected training point (the "maximin" criterion).
4. Continue until the desired training-set size is reached.

The test set is whatever was not selected.

The result is a training set that spans the convex hull of the data — training points
are spread throughout the feature space rather than clustered in any region.

## When to use it

- **Small to medium datasets** (up to a few thousand samples) where random splits
  could, by chance, concentrate training and test points in the same region.
- **Calibration and chemometrics** — Kennard–Stone is the de facto standard for
  NIR, Raman, and other spectroscopic calibration splits.
- **Regression** when you want to avoid interpolation bias: test points near training
  points look easy even for a bad model.
- **No target variable** is needed — the split is based on X only. If you have y and
  want to cover both, use [`SPXYSplit`](@ref) instead.

## Usage

```julia
using DataSplits

# Default Euclidean metric.
res = partition(X, KennardStoneSplit(); train = 0.8, test = 0.2)
X_train, X_test = splitdata(res, X)

# Custom metric.
using Distances
res = partition(X, KennardStoneSplit(Cityblock()); train = 70, test = 30)

# Large dataset — on-the-fly distances, O(N) memory.
res = partition(X, LazyKennardStoneSplit(); train = 0.8, test = 0.2)

# Train / validation / test.
res = partition(X, KennardStoneSplit(), KennardStoneSplit();
                train = 70, validation = 10, test = 20)
X_tr, X_val, X_te = splitdata(res, X)
```

## Eager vs. lazy

| | [`KennardStoneSplit`](@ref) | [`LazyKennardStoneSplit`](@ref) |
| --- | --- | --- |
| Distance matrix | Precomputed once | Computed on-the-fly each step |
| Memory | O(N²) | O(N) |
| Speed | Faster (matrix lookups) | Slightly slower (repeated distances) |
| Use when | Dataset fits in memory | N is large or memory is scarce |

## Limitations

- **Quadratic memory and time** for the eager variant — impractical for N > ~5000
  without the lazy variant.
- **Deterministic** — there is no randomness, so you cannot average over multiple
  Kennard–Stone splits. Use [`MoraisLimaMartinSplit`](@ref) if you need stochastic
  variants.
- **Feature scale matters** — if features have very different ranges the Euclidean
  metric will be dominated by high-variance features. Standardise X before splitting,
  or use a scaled metric (e.g. `WeightedEuclidean`).

## API reference

- [`KennardStoneSplit`](@ref) / [`CADEXSplit`](@ref)
- [`LazyKennardStoneSplit`](@ref) / [`LazyCADEXSplit`](@ref)

## References

Kennard, R. W.; Stone, L. A. Computer Aided Design of Experiments. *Technometrics*
1969, 11(1), 137–148. <https://doi.org/10.1080/00401706.1969.10490666>.
