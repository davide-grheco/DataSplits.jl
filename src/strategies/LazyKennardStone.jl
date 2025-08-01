using Distances

export LazyKennardStoneSplit, CADEXSplit

"""
    LazyKennardStoneSplit{T} <: SplitStrategy

Memory-efficient Kennard-Stone (CADEX) splitting strategy for large datasets.

Performs train/test splitting using the maximin strategy, but avoids storing the full NxN distance matrix in memory (O(N) storage).

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `metric::Distances.SemiMetric`: Distance metric to use (default: Euclidean())

# Examples
```julia
splitter = LazyKennardStoneSplit(0.8)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```
"""
struct LazyKennardStoneSplit{T} <: SplitStrategy
  frac::ValidFraction{T}
  metric::Distances.SemiMetric
end

# Constructor overloads
LazyKennardStoneSplit(frac::Real) = LazyKennardStoneSplit(ValidFraction(frac), Euclidean())
LazyKennardStoneSplit(frac::Real, metric) =
  LazyKennardStoneSplit(ValidFraction(frac), metric)
const LazyCADEXSplit = LazyKennardStoneSplit


"""
    _split(data, s::LazyKennardStoneSplit; rng=Random.GLOBAL_RNG) → (train_idx, test_idx)

Kennard-Stone (CADEX) algorithm for optimal train/test splitting using maximin strategy.
Memory-optimized implementation with O(N) storage.
Useful when working with large datasets where the NxN distance matrix does not fit memory.
When working with small datasets, use the traditional implementation.
"""
function _split(data, s::LazyKennardStoneSplit; rng = Random.GLOBAL_RNG)
  N = numobs(data)
  n_train, n_test = train_test_counts(N, s.frac)

  i₁, i₂ = find_most_distant_pair(data, s.metric)
  selected = falses(N)
  selected[i₁] = selected[i₂] = true
  order = Vector{Int}(undef, N)
  order[1:2] = [i₁, i₂]
  min_dists = fill(Inf, N)
  for i = 1:N
    if !selected[i]
      x = getobs(data, i)
      d1 = evaluate(s.metric, x, getobs(data, i₁))
      d2 = evaluate(s.metric, x, getobs(data, i₂))
      min_dists[i] = min(d1, d2)
    end
  end
  min_dists[i₁] = min_dists[i₂] = -Inf
  for k = 3:N
    next_i = argmax(min_dists)
    order[k] = next_i
    selected[next_i] = true
    min_dists[next_i] = -Inf
    ref = getobs(data, next_i)
    @inbounds for i = 1:N
      if !selected[i]
        d = evaluate(s.metric, ref, getobs(data, i))
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
    x = getobs(data, i)
    for j = (i+1):n
      y = getobs(data, j)
      d = evaluate(metric, x, y)
      if d > max_d
        max_d, best_i, best_j = d, i, j
      end
    end
  end

  return best_i, best_j
end
