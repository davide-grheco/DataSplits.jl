using Distances

export LazyKennardStoneSplit, CADEXSplit

struct LazyKennardStoneSplit{T} <: SplitStrategy
  frac::ValidFraction{T}
  metric::Distances.SemiMetric
end

# Constructor overloads
LazyKennardStoneSplit(frac::Real) = LazyKennardStoneSplit(ValidFraction(frac), Euclidean())
LazyKennardStoneSplit(frac::Real, metric) =
  LazyKennardStoneSplit(ValidFraction(frac), metric)
const CADEXSplit = LazyKennardStoneSplit  # Alias


"""
    _split(data, s::KennardStoneSplit; rng=Random.GLOBAL_RNG) → (train_idx, test_idx)

Kennard-Stone (CADEX) algorithm for optimal train/test splitting using maximin strategy.
Memory-optimized implementation with O(N) storage.
Useful when working with large datasets where the NxN distance matrix does not fit memory.
When working with small datasets, use the traditional implementation.
"""
function _split(data, s::LazyKennardStoneSplit; rng = Random.GLOBAL_RNG)
  idx_range = axes(data, 1)
  N = length(idx_range)
  n_test = round(Int, (1 - s.frac) * N)
  n_train = N - n_test
  idx_range = axes(data, 1)
  i₁, i₂ = find_most_distant_pair(data, idx_range; s.metric)

  if n_test < 2 || n_train < 2
    throw(
      ArgumentError(
        "Invalid split sizes: n_test=$n_test, n_train=$n_train. " *
        "Kennard-Stone requires at least 2 samples in the smallest split." *
        "Try changing your split fraction or check you are actually introducing enough data. ",
      ),
    )
  end


  if n_test <= n_train
    target_size = n_test
    select_test = true
  else
    target_size = n_train
    select_test = false
  end


  selected = falses(N)
  selected[i₁] = selected[i₂] = true
  selected_idx = [i₁, i₂]
  min_dists = Vector{float(eltype(first(data)))}(undef, N)
  fill!(min_dists, Inf)
  min_dists[i₁] = min_dists[i₂] = -Inf

  while length(selected_idx) != target_size
    # Update distances to last added point
    new_point = _get_sample(data, selected_idx[end])
    for i in eachindex(idx_range)
      if !selected[i]
        d = evaluate(s.metric, new_point, _get_sample(data, i))
        min_dists[i] = min(min_dists[i], d)
      end
    end

    # Select next point
    k = argmax(min_dists)
    push!(selected_idx, k)
    selected[k] = true
    min_dists[k] = -Inf
  end

  if select_test
    selected_idx = selected_idx
    train_idx = findall(!, selected)
  else
    train_idx = selected_idx
    selected_idx = findall(!, selected)
  end

  return _sort_indices!(idx_range, train_idx), _sort_indices!(idx_range, selected_idx)

end


function find_most_distant_pair(data, idx_vec; metric)
  max_d, i₁, i₂ = -Inf, 0, 0
  for i = 1:length(idx_vec)-1, j = i+1:length(idx_vec)
    x, y = data[idx_vec[i]], data[idx_vec[j]]
    d = evaluate(metric, x, y)
    if d > max_d
      max_d, i₁, i₂ = d, i, j
    end
  end
  return i₁, i₂
end

function nn_distance(A, B; metric)
  [minimum(evaluate.(Ref(metric), b, A)) for b in B]
end
