using Statistics: cov
using Distances

"""
    SPXYSplit(frac; metric_X=Euclidean(), metric_y=Euclidean())

Sample set Partitioning based on joint X–Y distance (SPXY).

A variant of Kennard–Stone where the joint distance matrix is the element-wise
sum of the (normalised) pairwise distance matrices of `X` and `y`.

# Fields
- `frac::ValidFraction`: Fraction of samples in the training subset (0 < frac < 1)
- `metric_X::Distances.SemiMetric`: Distance metric for `X` (default: `Euclidean()`)
- `metric_y::Distances.SemiMetric`: Distance metric for `y` (default: `Euclidean()`)

# Examples
```julia
res = partition(X, SPXYSplit(0.7); target=y)
res = partition(X, SPXYSplit(0.7; metric_X=Mahalanobis(cov(X; dims=2))); target=y)
X_train, X_test = splitdata(res, X)
```

# See also
[`KennardStoneSplit`](@ref) — the classical variant that uses only `X`.
"""
struct SPXYSplit <: AbstractSplitStrategy
  frac::ValidFraction
  metric_X::Distances.SemiMetric
  metric_y::Distances.SemiMetric
end

SPXYSplit(frac::Real; metric_X = Euclidean(), metric_y = Euclidean()) =
  SPXYSplit(ValidFraction(frac), metric_X, metric_y)

consumes(::SPXYSplit) = (:data, :target)
fallback_from_data(::SPXYSplit) = ()

"""
    MDKSSplit(frac; metric=nothing)

Minimum Dissimilarity Kennard–Stone (MDKS) split using Mahalanobis distance for `X`
and Euclidean distance for `y`.

If `metric` is not provided, the Mahalanobis distance is computed from the covariance
matrix of `X` at split time.

# Examples
```julia
res = partition(X, MDKSSplit(0.7); target=y)
res = partition(X, MDKSSplit(0.7; metric=Mahalanobis(cov(X; dims=2))); target=y)
X_train, X_test = splitdata(res, X)
```

# See also
[`SPXYSplit`](@ref)
"""
struct MDKSSplit <: AbstractSplitStrategy
  frac::ValidFraction
  metric::Union{Nothing,PreMetric}
end

MDKSSplit(frac::Real; metric = nothing) = MDKSSplit(ValidFraction(frac), metric)

consumes(::MDKSSplit) = (:data, :target)
fallback_from_data(::MDKSSplit) = ()

@inline function _norm_pairwise(X, metric)
  D = distance_matrix(X, metric)
  return D ./ maximum(D)
end

function _partition(X, s::SPXYSplit; target, rng = Random.GLOBAL_RNG, kwargs...)
  N = numobs(X)
  n_train, _ = train_test_counts(N, s.frac)
  DX = _norm_pairwise(X, s.metric_X)
  DY = _norm_pairwise(target, s.metric_y)
  DX .+= DY
  train_idx, test_idx = kennard_stone_from_distance_matrix(DX, n_train)
  return TrainTestSplit(train_idx, test_idx)
end

function _partition(X, s::MDKSSplit; target, rng = Random.GLOBAL_RNG, kwargs...)
  metric_X = s.metric === nothing ? Mahalanobis(cov(X; dims = 2)) : s.metric
  _partition(X, SPXYSplit(s.frac, metric_X, Euclidean()); target = target, rng = rng)
end
