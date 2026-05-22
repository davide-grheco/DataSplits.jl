```@meta
CurrentModule = DataSplits
```

# SPXY and MDKS

SPXY (Galvão et al. 2005) extends Kennard–Stone to **jointly cover both the feature
space and the target range**. It is the method of choice for regression tasks where
training-set diversity in y matters as much as diversity in X.

## How it works

1. Compute the pairwise distance matrix for X (features), normalised to [0, 1].
2. Compute the pairwise distance matrix for y (target), normalised to [0, 1].
3. Add the two matrices element-wise to form a **joint distance matrix**.
4. Apply the Kennard–Stone maximin selection on this joint matrix.

The normalisation ensures that X and y contribute equally regardless of their
absolute scales. The result is a training set that is simultaneously spread in
feature space and in response space.

## MDKS — Mahalanobis variant

MDKS (Saptoro et al. 2012) replaces the Euclidean distance for X with the
Mahalanobis distance, which accounts for correlations and different variances across
features. In practice MDKS often produces more useful splits when features are
correlated (e.g. spectroscopic data with overlapping absorption bands).

```julia
using DataSplits, Statistics

res = partition(X, MDKSSplit(); target = y, train = 0.8, test = 0.2)
```

If you have the covariance matrix precomputed:

```julia
using Distances
C = cov(X; dims = 2)
res = partition(X, MDKSSplit(; metric = Mahalanobis(C)); target = y, train = 0.8, test = 0.2)
```

## Usage

```julia
using DataSplits

# Default: Euclidean for both X and y.
res = partition(X, SPXYSplit(); target = y, train = 0.8, test = 0.2)
X_train, X_test = splitdata(res, X)
y_train, y_test = splitdata(res, y)

# Custom metrics.
using Distances
res = partition(X, SPXYSplit(; metric_X = Cityblock(), metric_y = Euclidean());
                target = y, train = 80, test = 20)

# Avoid storing two N×N matrices — but 32–57× slower.
res = partition(X, LazySPXYSplit(); target = y, train = 0.8, test = 0.2)
res = partition(X, LazyMDKSSplit(); target = y, train = 0.8, test = 0.2)
```

The `target` keyword is **required** — SPXY cannot fall back to `data` because `X`
and `y` are separate inputs.

## Eager vs. lazy

| | [`SPXYSplit`](@ref) / [`MDKSSplit`](@ref) | [`LazySPXYSplit`](@ref) / [`LazyMDKSSplit`](@ref) |
| --- | --- | --- |
| Peak memory | O(N²) — two full N×N matrices | O(N) — no full matrix |
| Speed (N=1000) | 14 ms / 12 ms | 445 ms / 708 ms (~32–57× slower) |

## When SPXY beats Kennard–Stone

Random split or Kennard–Stone can produce a training set that covers X well but has
all the extreme y values concentrated in test. If your model is a calibration curve,
this means the training regression extrapolates and performs poorly. SPXY prevents
this by including y in the diversity criterion.

## Limitations

- **Regression-focused** — for classification targets the normalised y-distance is
  not meaningful. Use [`StratifiedKFold`](@ref) or [`StratifiedShuffleSplit`](@ref)
  for classification.
- **y must be 1D** — multi-output regression is not directly supported.
- **Feature scale matters** — standardise X if features have very different ranges.

## API reference

- [`SPXYSplit`](@ref), [`LazySPXYSplit`](@ref)
- [`MDKSSplit`](@ref), [`LazyMDKSSplit`](@ref)

## References

Galvão, R. K. H. et al. A Method for Calibration and Validation Subset Partitioning.
*Talanta* 2005, 67(4), 736–740. <https://doi.org/10.1016/j.talanta.2005.03.025>.

Saptoro, A.; Tadé, M. O.; Vuthaluru, H. A Modified Kennard-Stone Algorithm for
Optimal Division of Data for Developing Artificial Neural Network Models.
*Chemical Product and Process Modeling* 2012, 7(1).
<https://doi.org/10.1515/1934-2659.1645>.
