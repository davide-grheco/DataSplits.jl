using Statistics: cov
using Distances


struct SPXYSplit <: SplitStrategy
  frac::ValidFraction
  metric_X::Distances.SemiMetric
  metric_y::Distances.SemiMetric
end

"""
    SPXYSplit(frac; metric_X = Euclidean(), metric_y = Euclidean())
    SPXYSplit(frac; metric = Euclidean())

Sample set Partitioning based on joint X–Y distance (SPXY).

Creates an SPXY splitter, a variant of Kennard–Stone in which the distance matrix is the element-wise sum of:
- the (normalized) pairwise distance matrix of the feature matrix `X` (using `metric_X`)
- plus the (normalized) pairwise distance matrix of the response vector `y` (using `metric_y`).

# Fields
- `frac::ValidFraction{T}`: Fraction of samples in the training subset (0 < frac < 1)
- `metric_X::Distances.SemiMetric`: Distance metric for `X` (default: Euclidean())
- `metric_y::Distances.SemiMetric`: Distance metric for `y` (default: Euclidean())

# Notes
- `split` **must** be called with a 2-tuple `(X, y)` or with positional arguments `split(X, y, strategy)`; calling `split(X, strategy)` will raise a `MethodError`.

# Examples
```julia
splitter = SPXYSplit(0.7)
splitter = SPXYSplit(0.7; metric_X=Mahalanobis(cov(X; dims=2)), metric_y=Euclidean())
result = split(X, y, splitter)
X_train, X_test = splitdata(result, X)
```

# See also
[`KennardStoneSplit`](@ref) — the classical variant that uses only `X`.
"""
SPXYSplit(frac::Real; metric_X = Euclidean(), metric_y = Euclidean()) =
  SPXYSplit(ValidFraction(frac), metric_X, metric_y)


struct MDKSSplit <: SplitStrategy
  frac::ValidFraction
  metric::Union{Nothing,PreMetric}
end


"""
    MDKSSplit(frac::Real; metric=nothing)

Minimum Dissimilarity Kennard–Stone (MDKS) split using the Mahalanobis distance.

If `metric` is not provided, the Mahalanobis distance is computed using the covariance matrix of `X`.

# Arguments
- `frac`: Fraction of samples to use for training (0 < frac < 1)
- `metric`: Optional distance metric. If not provided, Mahalanobis is computed from the data.

# Examples
```julia
splitter = MDKSSplit(0.7)  # auto-computes Mahalanobis from X
splitter = MDKSSplit(0.7; metric=Mahalanobis(cov(X; dims=2)))
result = split((X, y), splitter)
X_train, X_test = splitdata(result, X)
```

# See also
[`SPXYSplit`](@ref)
"""

MDKSSplit(frac::Real; metric = nothing) = MDKSSplit(ValidFraction(frac), metric)

function _split(X, y, s::MDKSSplit; rng = Random.GLOBAL_RNG)
  metric_X = s.metric === nothing ? Mahalanobis(cov(X; dims = 2)) : s.metric
  metric_y = Euclidean()
  spxy = SPXYSplit(s.frac, metric_X, metric_y)
  return _split(X, y, spxy; rng)
end

_split((X, y), s::MDKSSplit; kwargs...) = _split(X, y, s; kwargs...)

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
  N = numobs(X)

  n_train, n_test = train_test_counts(N, strategy.frac)
  DX = _norm_pairwise(X, strategy.metric_X)
  DY = _norm_pairwise(y, strategy.metric_y)
  D = DX .+ DY

  train_idx, test_idx = kennard_stone_from_distance_matrix(D, n_train)

  return TrainTestSplit(train_idx, test_idx)
end

# convenience method when the caller passes a tuple
_split((X, y), strategy::SPXYSplit; kwargs...) = _split(X, y, strategy; kwargs...)
