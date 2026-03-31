using Distances, Random

"""
    OptiSimSplit(frac; max_subsample_size=10, distance_cutoff=0.35, metric=Euclidean())

OptiSim (Clark 1997) K-dissimilarity selection strategy for train/test splitting.

# Fields
- `frac::ValidFraction`: Fraction of samples in the training subset (0 < frac < 1)
- `max_subsample_size::Integer`: Size of the temporary candidate subsample
- `distance_cutoff::Real`: Two points are "similar" if their distance < `distance_cutoff`
- `metric::Distances.SemiMetric`: Distance metric (default: `Euclidean()`)

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding
  Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.

# Examples
```julia
res = partition(X, OptiSimSplit(0.7; max_subsample_size=10))
X_train, X_test = splitdata(res, X)
```
"""
struct OptiSimSplit <: AbstractSplitStrategy
  frac::ValidFraction
  max_subsample_size::Integer
  distance_cutoff::Real
  metric::Distances.SemiMetric
end

function OptiSimSplit(
  frac::Real;
  max_subsample_size = 10,
  distance_cutoff = 0.35,
  metric = Euclidean(),
)
  OptiSimSplit(ValidFraction(frac), max_subsample_size, distance_cutoff, metric)
end

function OptiSimSplit(
  frac::ValidFraction;
  max_subsample_size = 10,
  distance_cutoff = 0.35,
  metric = Euclidean(),
)
  OptiSimSplit(frac, max_subsample_size, distance_cutoff, metric)
end

consumes(::OptiSimSplit) = (:data,)
fallback_from_data(::OptiSimSplit) = ()

function _partition(X, s::OptiSimSplit; rng = Random.GLOBAL_RNG, kwargs...)
  N = numobs(X)
  n_train, _ = train_test_counts(N, s.frac)
  D = distance_matrix(X, s.metric)
  selected_positions =
    optisim(D, n_train, s.max_subsample_size, s.distance_cutoff; rng = rng)
  train_pos = collect(selected_positions)
  test_pos = setdiff(1:N, train_pos)
  return TrainTestSplit(train_pos, test_pos)
end

function optisim(
  D::AbstractMatrix,
  selected_samples::Int = 10,
  max_subsample_size::Int = 0,
  distance_cutoff::Real = 0.35;
  rng = Random.GLOBAL_RNG,
)
  N = size(D, 1)
  M = min(selected_samples, N)
  K = max_subsample_size
  candidates = Set(1:N)
  selected = Set{Int}()
  first_selected = rand(rng, candidates)
  push!(selected, first_selected)
  delete!(candidates, first_selected)
  while length(selected) < M
    subsamples = _build_optisim_subsample!(D, selected, candidates, K, distance_cutoff, rng)
    if !isempty(subsamples)
      best = find_maximin_element(D, subsamples, selected)
      push!(selected, best)
      delete!(candidates, best)
    else
      break
    end
  end
  return selected
end

function _build_optisim_subsample!(
  D::Matrix{Float64},
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
    is_dissimilar = all(D[candidate, s] >= min_dissimilarity for s in selected)
    if is_dissimilar
      push!(subsample, candidate)
    else
      delete!(candidates, candidate)
    end
  end
  return subsample
end
