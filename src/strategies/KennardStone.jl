using Distances, MLUtils

"""
    KennardStoneSplit <: AbstractSplitStrategy

In-memory Kennard-Stone (CADEX) algorithm for train/test splitting.

Precomputes the full N×N distance matrix; prefer `LazyKennardStoneSplit`
for large datasets where that is prohibitive.

# Fields
- `metric::Distances.SemiMetric`: Distance metric (default: `Euclidean()`)

# Examples
```julia
res = partition(X, KennardStoneSplit(); train = 80, test = 20)
X_train, X_test = splitdata(res, X)

res = partition(X, KennardStoneSplit(Cityblock()); train = 70, test = 30)
```
"""
struct KennardStoneSplit <: AbstractSplitStrategy
  metric::Distances.SemiMetric
end

KennardStoneSplit() = KennardStoneSplit(Euclidean())

const CADEXSplit = KennardStoneSplit

consumes(::KennardStoneSplit) = (:data,)
fallback_from_data(::KennardStoneSplit) = ()

function _partition(
  data,
  s::KennardStoneSplit;
  n_train,
  n_test,
  rng = Random.GLOBAL_RNG,
  kwargs...,
)
  D = distance_matrix(data, s.metric)
  train_pos, test_pos = kennard_stone_from_distance_matrix(D, n_train)
  return TrainTestSplit(train_pos, test_pos)
end

function _partition(
  data::AbstractVector{<:AbstractVector},
  s::KennardStoneSplit;
  n_train,
  n_test,
  rng = Random.GLOBAL_RNG,
  kwargs...,
)
  _partition(stack(data), s; n_train, n_test, rng)
end

"""
    find_most_distant_pair(D::AbstractMatrix) -> (i, j)

Find the indices of the most distant pair in a precomputed distance matrix.
"""
function find_most_distant_pair(D::AbstractMatrix)
  n = size(D, 1)
  max_d = -Inf
  i₁, i₂ = 1, 2
  @inbounds for i = 1:(n-1)
    for j = (i+1):n
      if D[i, j] > max_d
        max_d = D[i, j]
        i₁, i₂ = i, j
      end
    end
  end
  return i₁, i₂
end

function kennard_stone_from_distance_matrix(D::AbstractMatrix, n_train::Integer)
  N = size(D, 1)
  selected = falses(N)
  i₁, i₂ = find_most_distant_pair(D)
  selected[i₁] = selected[i₂] = true
  selected_count = 2
  selected_order = Vector{Int}(undef, N)
  selected_order[1:2] = [i₁, i₂]
  min_dists = min.(view(D, :, i₁), view(D, :, i₂))
  min_dists[i₁] = min_dists[i₂] = -Inf
  while selected_count < N
    k_idx = argmax(min_dists)
    selected_count += 1
    selected[k_idx] = true
    selected_order[selected_count] = k_idx
    min_dists .= min.(min_dists, view(D, :, k_idx))
    min_dists[k_idx] = -Inf
  end
  train_idx = selected_order[1:n_train]
  test_idx = selected_order[(n_train+1):end]
  return train_idx, test_idx
end
