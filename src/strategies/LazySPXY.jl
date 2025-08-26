using Distances
using MLUtils
import MLUtils: getobs, numobs


struct LazySPXYSplit <: SplitStrategy
  frac::ValidFraction
  metric_X::Distances.SemiMetric
  metric_y::Distances.SemiMetric
end

"""
    LazySPXYSplit{T} <: SplitStrategy

Memory-efficient SPXY splitting strategy for large datasets.
Performs train/test splitting using the maximin strategy, but avoids storing the full NxN distance matrix in memory (O(N) storage).

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `metric_X::Distances.SemiMetric`: Distance metric for X (default: Euclidean())
- `metric_y::Distances.SemiMetric`: Distance metric for y (default: Euclidean())

# Examples
```julia
splitter = LazySPXYSplit(0.8)
result = split((X, y), splitter)
X_train, X_test = splitdata(result, X)
```
"""
LazySPXYSplit(frac::Real; metric_X = Euclidean(), metric_y = Euclidean()) =
  LazySPXYSplit(ValidFraction(frac), metric_X, metric_y)


function _find_max_distance_XY(X, y, metric_X, metric_y)
  N = numobs(X)
  max_dx = 0.0
  max_dy = 0.0
  for i = 1:(N-1)
    xi, yi = X[i, :], y[i]
    for j = (i+1):N
      xj, yj = X[j, :], y[j]
      dx = evaluate(metric_X, xi, xj)
      dy = evaluate(metric_y, yi, yj)
      max_dx = max(max_dx, dx)
      max_dy = max(max_dy, dy)
    end
  end
  return max_dx, max_dy
end

struct XYObsTable
  X::Any
  y::Any
end

getobs(data::XYObsTable, i) = (getobs(data.X, i), getobs(data.y, i))
numobs(data::XYObsTable) = numobs(data.X)

struct LazySPXYMetric <: Distances.SemiMetric
  metric_X::Distances.SemiMetric
  metric_y::Distances.SemiMetric
  max_X::Float64
  max_y::Float64
end

function Distances.evaluate(m::LazySPXYMetric, a, b)
  x1, y1 = a
  x2, y2 = b
  dx = Distances.evaluate(m.metric_X, x1, x2) / m.max_X
  dy = Distances.evaluate(m.metric_y, y1, y2) / m.max_y
  return dx + dy
end

function _split((X, y), s::LazySPXYSplit; rng = Random.GLOBAL_RNG)
  max_X, max_y = _find_max_distance_XY(X, y, s.metric_X, s.metric_y)
  if max_X == 0.0
    max_X = 1.0
  end
  if max_y == 0.0
    max_y = 1.0
  end
  metric = LazySPXYMetric(s.metric_X, s.metric_y, max_X, max_y)
  data = XYObsTable(X, y)
  return _split(data, LazyKennardStoneSplit(s.frac, metric); rng)
end


struct LazyMDKSSplit <: SplitStrategy
  frac::ValidFraction
  metric_X::Union{Nothing,Distances.SemiMetric}
  metric_y::Distances.SemiMetric
end

"""
    LazyMDKSSplit <: SplitStrategy

Memory-efficient Minimum Dissimilarity Kennard–Stone (MDKS) splitting strategy for large datasets.
Performs train/test splitting using the maximin strategy, but avoids storing the full NxN distance matrix in memory (O(N) storage).
Uses Mahalanobis distance for X and Euclidean for y, normalized and summed as in SPXY.

# Fields
- `frac::ValidFraction`: Fraction of data to use for training (0 < frac < 1)
- `metric_X::Union{Nothing,Distances.SemiMetric}`: Distance metric for X (default: Mahalanobis(cov(X; dims=2)))
- `metric_y::Distances.SemiMetric`: Distance metric for y (default: Euclidean())

# Examples
```julia
splitter = LazyMDKSSplit(0.7)
result = split((X, y), splitter)
X_train, X_test = splitdata(result, X)
```
"""
function LazyMDKSSplit(frac::Real; metric = nothing)
  return LazyMDKSSplit(
    ValidFraction(frac),
    metric === nothing ? nothing : metric,
    Euclidean(),
  )
end

function _split((X, y), s::LazyMDKSSplit; rng = Random.GLOBAL_RNG)
  metric_X = s.metric_X === nothing ? Mahalanobis(cov(X; dims = 2)) : s.metric_X
  metric_y = s.metric_y
  max_X, max_y = _find_max_distance_XY(X, y, metric_X, metric_y)
  if max_X == 0.0
    max_X = 1.0
  end
  if max_y == 0.0
    max_y = 1.0
  end
  metric = LazySPXYMetric(metric_X, metric_y, max_X, max_y)
  data = XYObsTable(X, y)
  return _split(data, LazyKennardStoneSplit(s.frac, metric); rng)
end
