using Distances
using MLUtils
import MLUtils: getobs, numobs

"""
    LazySPXYSplit <: AbstractSplitStrategy

Memory-efficient SPXY splitting strategy. Computes distances on-the-fly
(O(N) storage) rather than precomputing the full N×N matrix.

# Fields
- `metric_X::Distances.SemiMetric`: Distance metric for `X` (default: `Euclidean()`)
- `metric_y::Distances.SemiMetric`: Distance metric for `y` (default: `Euclidean()`)

# Examples
```julia
res = partition(X, LazySPXYSplit(); target=y, train=80, test=20)
X_train, X_test = splitdata(res, X)
```
"""
struct LazySPXYSplit <: AbstractSplitStrategy
  metric_X::Distances.SemiMetric
  metric_y::Distances.SemiMetric
end

LazySPXYSplit(; metric_X = Euclidean(), metric_y = Euclidean()) =
  LazySPXYSplit(metric_X, metric_y)

consumes(::LazySPXYSplit) = (:data, :target)
fallback_from_data(::LazySPXYSplit) = ()

"""
    LazyMDKSSplit <: AbstractSplitStrategy

Memory-efficient Minimum Dissimilarity Kennard–Stone (MDKS) splitting strategy.
Uses Mahalanobis distance for `X` and Euclidean for `y`, normalised and summed
as in SPXY. Computes distances on-the-fly (O(N) storage).

# Fields
- `metric_X::Union{Nothing,Distances.SemiMetric}`: Distance metric for `X`;
  if `nothing`, Mahalanobis is computed from the data at split time.
- `metric_y::Distances.SemiMetric`: Distance metric for `y` (default: `Euclidean()`)

# Examples
```julia
res = partition(X, LazyMDKSSplit(); target=y, train=70, test=30)
X_train, X_test = splitdata(res, X)
```
"""
struct LazyMDKSSplit <: AbstractSplitStrategy
  metric_X::Union{Nothing,Distances.SemiMetric}
  metric_y::Distances.SemiMetric
end

LazyMDKSSplit(; metric = nothing) = LazyMDKSSplit(metric, Euclidean())

consumes(::LazyMDKSSplit) = (:data, :target)
fallback_from_data(::LazyMDKSSplit) = ()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

function _find_max_distance_XY(X, y, metric_X, metric_y)
  N = numobs(X)
  max_dx = 0.0
  max_dy = 0.0
  for i = 1:(N-1)
    xi = getobs(X, i)
    yi = y[i]
    for j = (i+1):N
      xj = getobs(X, j)
      yj = y[j]
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

# ---------------------------------------------------------------------------
# _partition implementations
# ---------------------------------------------------------------------------

function _partition(
  X,
  s::LazySPXYSplit;
  target,
  n_train,
  n_test,
  rng = Random.GLOBAL_RNG,
  kwargs...,
)
  max_X, max_y = _find_max_distance_XY(X, target, s.metric_X, s.metric_y)
  max_X = max_X == 0.0 ? 1.0 : max_X
  max_y = max_y == 0.0 ? 1.0 : max_y
  metric = LazySPXYMetric(s.metric_X, s.metric_y, max_X, max_y)
  data = XYObsTable(X, target)
  return _partition(
    data,
    LazyKennardStoneSplit(metric);
    n_train = n_train,
    n_test = n_test,
    rng = rng,
  )
end

function _partition(
  X,
  s::LazyMDKSSplit;
  target,
  n_train,
  n_test,
  rng = Random.GLOBAL_RNG,
  kwargs...,
)
  metric_X = s.metric_X === nothing ? Mahalanobis(cov(X; dims = 2)) : s.metric_X
  max_X, max_y = _find_max_distance_XY(X, target, metric_X, s.metric_y)
  max_X = max_X == 0.0 ? 1.0 : max_X
  max_y = max_y == 0.0 ? 1.0 : max_y
  metric = LazySPXYMetric(metric_X, s.metric_y, max_X, max_y)
  data = XYObsTable(X, target)
  return _partition(
    data,
    LazyKennardStoneSplit(metric);
    n_train = n_train,
    n_test = n_test,
    rng = rng,
  )
end
