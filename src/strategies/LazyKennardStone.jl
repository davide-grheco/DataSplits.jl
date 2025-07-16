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
const LazyCADEXSplit = LazyKennardStoneSplit


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

  if n_test < 2 || n_train < 2
    throw(
      ArgumentError(
        "Invalid split sizes: n_test=$n_test, n_train=$n_train. " *
        "Kennard-Stone requires at least 2 samples in the smallest split." *
        "Try changing your split fraction or check you are actually introducing enough data. ",
      ),
    )
  end

  i₁, i₂ = find_most_distant_pair(data, idx_range; s.metric)

  selected = falses(N)
  pos = Dict(idx_range .=> 1:N)
  selected[pos[i₁]] = selected[pos[i₂]] = true

  selected_order = Vector{eltype(idx_range)}(undef, N)
  selected_order[1:2] = [i₁, i₂]

  min_dists = fill(Inf, N)
  row_of(i) = _get_sample(data, i)
  for ax in idx_range
    p = pos[ax]
    if !selected[p]
      x = row_of(ax)
      d1 = evaluate(s.metric, x, row_of(i₁))
      d2 = evaluate(s.metric, x, row_of(i₂))
      min_dists[p] = min(d1, d2)
    end
  end
  min_dists[pos[i₁]] = min_dists[pos[i₂]] = -Inf

  for k = 3:N
    next_pos = argmax(min_dists)
    next_ax = idx_range[next_pos]

    selected_order[k] = next_ax
    selected[next_pos] = true
    min_dists[next_pos] = -Inf

    ref_vec = row_of(next_ax)
    @inbounds for p = 1:N
      if !selected[p]
        cand_ax = idx_range[p]
        d = evaluate(s.metric, ref_vec, row_of(cand_ax))
        min_dists[p] = min(min_dists[p], d)
      end
    end
  end

  train_idx = selected_order[1:n_train]
  test_idx = selected_order[n_train+1:end]

  return sort(train_idx), sort(test_idx)
end


function find_most_distant_pair(data, idx_vec; metric)
  max_d, i₁, i₂ = -Inf, 0, 0
  for i = 1:length(idx_vec)-1, j = i+1:length(idx_vec)
    x, y = _get_sample(data, idx_vec[i]), _get_sample(data, idx_vec[j])
    d = evaluate(metric, x, y)
    if d > max_d
      max_d, i₁, i₂ = d, idx_vec[i], idx_vec[j]
    end
  end
  return i₁, i₂
end

function nn_distance(A, B; metric)
  [minimum(evaluate.(Ref(metric), b, A)) for b in B]
end
