using Distances

struct SPXYSplit{T,M<:Distances.SemiMetric} <: SplitStrategy
  frac::ValidFraction{T}
  metric::M
end

"""
    SPXYSplit(frac; metric = Euclidean())

Sample set Partitioning based on joint X–Y distance (SPXY).

Creates an SPXY splitter, a variant of Kennard–Stone in which the distance matrix is the element-wise sum of:
- the (normalized) pairwise distance matrix of the feature matrix `X`
- plus the (normalized) pairwise distance matrix of the response vector `y`.

# Fields
- `frac::ValidFraction{T}`: Fraction of samples in the training subset (0 < frac < 1)
- `metric::Distances.SemiMetric`: Distance metric used for both `X` and `y` (default: Euclidean())

# Notes
- `split` **must** be called with a 2-tuple `(X, y)` or with positional arguments `split(X, y, strategy)`; calling `split(X, strategy)` will raise a `MethodError`.

# Examples
```julia
splitter = SPXYSplit(0.7)
result = split(X, y, splitter)
X_train, X_test = splitdata(result, X)
```

# See also
[`KennardStoneSplit`](@ref) — the classical variant that uses only `X`.
"""
SPXYSplit(frac::Real; metric = Euclidean()) = SPXYSplit(ValidFraction(frac), metric)

"""
    MDKSSplit(frac::Real)

Alias for `SPXYSplit(frac; metric = Mahalanobis())`. Uses the Mahalanobis distance for both X and y.

# Arguments
- `frac`: Fraction of samples to use for training (0 < frac < 1)

# Examples
```julia
splitter = MDKSSplit(0.7)
result = split((X, y), splitter)
X_train, X_test = splitdata(result, X)
```
"""
MDKSSplit(frac::Real) = SPXYSplit(frac; metric = Mahalanobis())

@inline function _norm_pairwise(X, metric)
  D = distance_matrix(X, metric)
  return D ./ maximum(D)
end


"""
    _split(X, y, strategy::SPXYSplit; rng = Random.GLOBAL_RNG) → (train_idx, test_idx)

Split a **feature matrix `X`** and **response vector `y`** into
train/test subsets using the SPXY algorithm:

1.  Build the joint distance matrix `D = D_X + D_Y`
    (see [`SPXYSplit`](@ref) for details).
2.  Run the Kennard–Stone maximin procedure on `D`.
3.  Return two **sorted** index vectors (`train_idx`, `test_idx`).

### Arguments
| name      | type                 | requirement |
|:--------- |:-------------------- |:----------- |
| `X`       | `AbstractMatrix`     | `size(X, 1) == length(y)` |
| `y`       | `AbstractVector`     |             |
| `strategy`| `SPXYSplit`          | created with `SPXYSplit(frac; metric)` |
| `rng`     | random‑number source | *unused* here but kept for API symmetry |

### Returns
Two `Vector{Int}` with the **sample indices** of `X` (and the corresponding
entries of `y`) that belong to the training and test subsets.
"""
function _split(
  X::AbstractArray,
  y::AbstractVector,
  strategy::SPXYSplit;
  rng = Random.GLOBAL_RNG,
)
  idx_range = axes(X, 1)
  N = length(idx_range)

  n_test = round(Int, (1 - strategy.frac) * N)
  n_train = N - n_test

  DX = _norm_pairwise(X, strategy.metric)
  DY = _norm_pairwise(y, strategy.metric)
  D = DX .+ DY

  train_idx, test_idx = kennard_stone_from_distance_matrix(D, n_train)

  return TrainTestSplit(train_idx, test_idx)
end

# convenience method when the caller passes a tuple
_split((X, y), strategy::SPXYSplit; kwargs...) = _split(X, y, strategy; kwargs...)
