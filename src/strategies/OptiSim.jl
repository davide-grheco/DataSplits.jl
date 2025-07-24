
using Distances, Random


"""
    OptiSimSplit(frac; n_clusters = 10, max_subsample_size, distance_cutoff = 0.10,
                 metric = Euclidean(), random_state = 42)

Implementation of **OptiSim** (Clark 1998, _J. Chem. Inf. Comput. Sci._),
an optimisable K‑dissimilarity selection.

* `frac` – fraction of samples to return in the **training** subset
* `n_clusters = M` – requested cluster/selection‑set size
* `max_subsample_size = K` – size of the temporary sub‑sample
  (default: `max(1, ceil(Int, 0.05N))`)
* `distance_cutoff = c` – two points are “similar” if their distance < `c`
* `metric` – any `Distances.jl` metric
* `random_state` – seed for the RNG

The splitter requires **both** an `X` matrix and target vector `y`
when calling `split`.
"""
struct OptiSimSplit{T,M<:Distances.SemiMetric} <: SplitStrategy
  frac::ValidFraction{T}
  max_subsample_size::Integer
  distance_cutoff::Real
  metric::M
end

function OptiSimSplit(
  frac::Real;
  max_subsample_size = 0,
  distance_cutoff = 0.35,
  metric = Euclidean(),
)

  OptiSimSplit(ValidFraction(frac), max_subsample_size, distance_cutoff, metric)
end

function optisim_split(N, s, rng, X)
  n_train = round(Int, s.frac * N)
  D = distance_matrix(X, s.metric)
  selected_positions =
    optisim(D, n_train, s.max_subsample_size, s.distance_cutoff; rng = rng)
  train_pos = collect(selected_positions)
  test_pos = setdiff(1:N, train_pos)
  return TrainTestSplit(train_pos, test_pos)
end

function _split(X, s::OptiSimSplit; rng = Random.GLOBAL_RNG)
  split_with_positions(X, s, optisim_split; rng = rng)
end


function optisim(
  distance_matrix::AbstractMatrix,
  selected_samples::Int = 10,
  max_subsample_size::Int = 0,
  distance_cutoff::Real = 0.35;
  rng = Random.GLOBAL_RNG,
)
  N = size(distance_matrix, 1)
  M = min(selected_samples, N)
  K = max_subsample_size

  rchoose(set) = rand(rng, collect(set))
  getdist(i, j) = i < j ? D[j, i] : D[i, j]

  candidates = Set(1:N)
  selected = Set{Int}()

  push!(selected, rchoose(candidates))
  delete!(candidates, selected)

  while length(selected) < M
    subsamples = _build_optisim_subsample(
      distance_matrix,
      selected,
      candidates,
      K,
      distance_cutoff,
      rng,
    )

    if !isempty(subsamples)
      best = find_maximin_element(distance_matrix, subsamples, selected)
      push!(selected, best)
      delete!(candidates, best)
    else
      break
    end
  end

  return selected
end

function _build_optisim_subsample(
  distance_matrix::Matrix{Float64},
  selected::Set{Int},
  candidates::Set{Int},
  subset_size::Int,
  min_dissimilarity::Real,
  rng::AbstractRNG,
)
  subsample = Set{Int}()
  remaining_candidates = copy(candidates)

  while length(subsample) < subset_size && !isempty(remaining_candidates)
    candidate = rand(rng, remaining_candidates)
    delete!(remaining_candidates, candidate)

    is_dissimilar =
      all(distance_matrix[candidate, s] >= min_dissimilarity for s in selected)

    # TODO: Consider deleting invalid samples when implementing the lazy version
    if is_dissimilar
      push!(subsample, candidate)
    end
  end

  return subsample
end
