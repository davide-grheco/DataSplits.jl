using Distances, MLUtils

export KennardStoneSplit, CADEXSplit

"""
    KennardStoneSplit{T} <: SplitStrategy

Splitting strategy implementing the Kennard-Stone algorithm for train/test splitting.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `metric::Distances.SemiMetric`: Distance metric to use (default: Euclidean())

# Examples
```julia
splitter = KennardStoneSplit(0.8)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)

# Custom metric
using Distances
splitter = KennardStoneSplit(0.7, Cityblock())
```
"""
struct KennardStoneSplit{T} <: SplitStrategy
  frac::ValidFraction{T}
  metric::Distances.SemiMetric
end

# Constructor overloads
KennardStoneSplit(frac::Real) = KennardStoneSplit(ValidFraction(frac), Euclidean())
KennardStoneSplit(frac::Real, metric) = KennardStoneSplit(ValidFraction(frac), metric)
const CADEXSplit = KennardStoneSplit  # Alias

"""
    _split(data, s::KennardStoneSplit; rng=Random.GLOBAL_RNG) → (train_idx, test_idx)

Optimized in-memory Kennard-Stone algorithm using precomputed distance matrix.
Best for small-to-medium datasets where O(N²) memory is acceptable.
"""
function _split(data, s::KennardStoneSplit; rng = Random.GLOBAL_RNG)
  N = numobs(data)
  n_train, n_test = train_test_counts(N, s.frac)

  D = distance_matrix(data, s.metric)
  train_pos, test_pos = kennard_stone_from_distance_matrix(D, n_train)
  return TrainTestSplit(train_pos, test_pos)
end

function _split(
  data::AbstractVector{<:AbstractVector},
  s::KennardStoneSplit;
  rng = Random.GLOBAL_RNG,
)
  data = stack(data)
  _split(data, s; rng)
end

"""
    find_most_distant_pair(D::AbstractMatrix) → (i, j)

Finds indices of most distant pair in precomputed distance matrix.
"""
function find_most_distant_pair(D::AbstractMatrix)
  n = size(D, 1)
  max_d = -Inf
  i₁, i₂ = 1, 2

  @inbounds for i = 1:n-1
    for j = i+1:n
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


  # TODO: This unnecessarily goes through all samples
  # It would be enough to go through n_train samples
  while selected_count < N
    k_idx = argmax(min_dists)
    selected_count += 1
    selected[k_idx] = true
    selected_order[selected_count] = k_idx
    min_dists .= min.(min_dists, view(D, :, k_idx))
    min_dists[k_idx] = -Inf
  end


  train_idx = selected_order[1:n_train]
  test_idx = selected_order[n_train+1:end]

  return train_idx, test_idx
end
