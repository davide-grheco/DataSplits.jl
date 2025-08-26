using Distances, Random, MLUtils
import MLUtils: getobs, numobs

"""
    LazyOptiSimSplit <: SplitStrategy

Memory-efficient, lazy implementation of the OptiSim (Clark 1997) dissimilarity selection strategy.
Avoids building the full NxN distance matrix by computing distances on-the-fly.

- `frac`: Fraction of samples to use for training (0 < frac < 1)
- `max_subsample_size`: Size of the temporary candidate subsample (default: 10)
- `distance_cutoff`: Two points are “similar” if their distance < `distance_cutoff` (default: 0.35)
- `metric`: Distance metric for X (default: Euclidean())

References:
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.
"""
struct LazyOptiSimSplit <: SplitStrategy
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

function _split(X, s::LazyOptiSimSplit; rng = Random.GLOBAL_RNG)
  N = numobs(X)
  n_train, _ = train_test_counts(N, s.frac)
  candidates = Set(1:N)
  selected = Set{Int}()

  # Randomly pick the first sample
  push!(selected, rand(rng, candidates))
  delete!(candidates, first(selected))

  while length(selected) < n_train && !isempty(candidates)
    # Build a random subsample of candidates that are sufficiently dissimilar
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
        # If the candidate is too similar to an element it will always be, we can discard it from further analysis
        delete!(candidates, candidate)
      end
    end

    if !isempty(subsample)
      # Find the candidate with the maximum minimum distance to the selected set
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
