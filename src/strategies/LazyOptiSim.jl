using Distances, Random, MLUtils
import MLUtils: getobs, numobs

"""
    LazyOptiSimSplit <: AbstractSplitStrategy

Memory-efficient, lazy implementation of the OptiSim (Clark 1997) dissimilarity
selection strategy. Computes distances on-the-fly; avoids the full N×N matrix.

# Fields
- `frac::ValidFraction`: Fraction of samples in the training subset (0 < frac < 1)
- `max_subsample_size::Int`: Size of the temporary candidate subsample (default: 10)
- `distance_cutoff::Float64`: Similarity threshold (default: 0.35)
- `metric::Distances.SemiMetric`: Distance metric (default: `Euclidean()`)

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding
  Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.
"""
struct LazyOptiSimSplit <: AbstractSplitStrategy
  frac::ValidFraction
  max_subsample_size::Int
  distance_cutoff::Float64
  metric::Distances.SemiMetric
end

function LazyOptiSimSplit(
  frac::Real;
  max_subsample_size = 10,
  distance_cutoff = 0.35,
  metric = Euclidean(),
)
  LazyOptiSimSplit(ValidFraction(frac), max_subsample_size, distance_cutoff, metric)
end

function LazyOptiSimSplit(
  frac::ValidFraction;
  max_subsample_size = 10,
  distance_cutoff = 0.35,
  metric = Euclidean(),
)
  LazyOptiSimSplit(frac, max_subsample_size, distance_cutoff, metric)
end

consumes(::LazyOptiSimSplit) = (:data,)
fallback_from_data(::LazyOptiSimSplit) = ()

function _partition(X, s::LazyOptiSimSplit; rng = Random.GLOBAL_RNG, kwargs...)
  N = numobs(X)
  n_train, _ = train_test_counts(N, s.frac)
  candidates = Set(1:N)
  selected = Set{Int}()
  push!(selected, rand(rng, candidates))
  delete!(candidates, first(selected))
  while length(selected) < n_train && !isempty(candidates)
    subsample = Set{Int}()
    remaining_candidates = copy(candidates)
    while length(subsample) < s.max_subsample_size && !isempty(remaining_candidates)
      candidate = rand(rng, remaining_candidates)
      delete!(remaining_candidates, candidate)
      value = getobs(X, candidate)
      is_dissimilar = all(
        sel -> Distances.evaluate(s.metric, value, getobs(X, sel)) >= s.distance_cutoff,
        selected,
      )
      if is_dissimilar
        push!(subsample, candidate)
      else
        delete!(candidates, candidate)
      end
    end
    if !isempty(subsample)
      best = find_maximin_element_lazy(X, s.metric, subsample, selected)
      push!(selected, best)
      delete!(candidates, best)
    else
      break
    end
  end
  train_pos = collect(selected)
  test_pos = setdiff(1:N, train_pos)
  return TrainTestSplit(train_pos, test_pos)
end
