using Distances, Random, MLUtils
import MLUtils: getobs, numobs

"""
    LazyOptiSimSplit <: AbstractSplitStrategy

Memory-efficient, lazy implementation of the OptiSim (Clark 1997) dissimilarity
selection strategy. Computes distances on-the-fly; avoids the full N×N matrix.

# Fields
- `max_subsample_size::Int`: Size of the temporary candidate subsample (default: 10)
- `distance_cutoff::Float64`: Similarity threshold (default: 0.35)
- `metric::Distances.SemiMetric`: Distance metric (default: `Euclidean()`)

# Notes
Emits the same undershoot warning as [`OptiSimSplit`](@ref) under
`_id = :datasplits_optisim_undershoot` when `distance_cutoff` exhausts the
candidate pool before reaching `n_train`. See `?OptiSimSplit` for the
silencing recipe.

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding
  Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.

# Examples
```julia
res = partition(X, LazyOptiSimSplit(); train = 70, test = 30)
X_train, X_test = splitdata(res, X)
```
"""
struct LazyOptiSimSplit <: AbstractSplitStrategy
  max_subsample_size::Int
  distance_cutoff::Float64
  metric::Distances.SemiMetric
end

function LazyOptiSimSplit(;
  max_subsample_size = 10,
  distance_cutoff = 0.35,
  metric = Euclidean(),
)
  max_subsample_size >= 0 || throw(
    SplitParameterError(
      "`max_subsample_size` must be non-negative, got $max_subsample_size.",
    ),
  )
  distance_cutoff >= 0 || throw(
    SplitParameterError("`distance_cutoff` must be non-negative, got $distance_cutoff."),
  )
  LazyOptiSimSplit(max_subsample_size, distance_cutoff, metric)
end

consumes(::LazyOptiSimSplit) = (:data,)
fallback_from_data(::LazyOptiSimSplit) = ()

function _partition(
  X,
  s::LazyOptiSimSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  N = numobs(X)
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
  _warn_undershoot(
    length(train_pos),
    n_train,
    "OptiSim: selected $(length(train_pos))/$n_train training samples; " *
    "distance_cutoff=$(s.distance_cutoff) exhausted the candidate pool. " *
    "Lower `distance_cutoff`, reduce `train`, or silence this warning by its " *
    "`_id` (see `?OptiSimSplit`).";
    id = :datasplits_optisim_undershoot,
  )
  return TrainTestSplit(train_pos, test_pos)
end
