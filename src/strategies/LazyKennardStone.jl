using Distances

"""
    LazyKennardStoneSplit <: AbstractSplitStrategy

Memory-efficient Kennard-Stone (CADEX) algorithm. Computes distances
on-the-fly (O(N) storage) rather than precomputing the full N×N matrix.

# Fields
- `metric::Distances.SemiMetric`: Distance metric (default: `Euclidean()`)

# Examples
```julia
res = partition(X, LazyKennardStoneSplit(); train = 80, test = 20)
X_train, X_test = splitdata(res, X)
```
"""
struct LazyKennardStoneSplit{M<:Distances.SemiMetric} <: AbstractSplitStrategy
  metric::M
end

LazyKennardStoneSplit() = LazyKennardStoneSplit(Euclidean())

"""
    LazyCADEXSplit

Alias for [`LazyKennardStoneSplit`](@ref).
"""
const LazyCADEXSplit = LazyKennardStoneSplit

consumes(::LazyKennardStoneSplit) = (:data,)
fallback_from_data(::LazyKennardStoneSplit) = ()

function _partition(
  data,
  s::LazyKennardStoneSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  N = numobs(data)
  i₁, i₂ = find_most_distant_pair(data, s.metric)
  selected = falses(N)
  selected[i₁] = selected[i₂] = true
  order = Vector{Int}(undef, N)
  order[1:2] = [i₁, i₂]
  min_dists = fill(Inf, N)
  for i = 1:N
    if !selected[i]
      x = _obs(data, i)
      d1 = Distances.evaluate(s.metric, x, _obs(data, i₁))
      d2 = Distances.evaluate(s.metric, x, _obs(data, i₂))
      min_dists[i] = min(d1, d2)
    end
  end
  min_dists[i₁] = min_dists[i₂] = -Inf
  for k = 3:N
    next_i = argmax(min_dists)
    order[k] = next_i
    selected[next_i] = true
    min_dists[next_i] = -Inf
    ref = _obs(data, next_i)
    @inbounds for i = 1:N
      if !selected[i]
        d = Distances.evaluate(s.metric, ref, _obs(data, i))
        min_dists[i] = min(min_dists[i], d)
      end
    end
  end
  train_pos = order[1:n_train]
  test_pos = order[(n_train+1):end]
  return TrainTestSplit(train_pos, test_pos)
end

function find_most_distant_pair(data, metric::Distances.SemiMetric)
  max_d, best_i, best_j = -Inf, nothing, nothing
  n = numobs(data)
  for i = 1:(n-1)
    x = _obs(data, i)
    for j = (i+1):n
      y = _obs(data, j)
      d = Distances.evaluate(metric, x, y)
      if d > max_d
        max_d, best_i, best_j = d, i, j
      end
    end
  end
  return best_i, best_j
end
