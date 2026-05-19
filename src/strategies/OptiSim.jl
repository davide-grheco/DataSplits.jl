using Distances, Random

"""
    OptiSimSplit(; max_subsample_size=10, distance_cutoff=0.35, metric=Euclidean())

OptiSim (Clark 1997) K-dissimilarity selection strategy for train/test splitting.

# Fields
- `max_subsample_size::Int`: Size of the temporary candidate subsample
- `distance_cutoff::Float64`: Two points are "similar" if their distance < `distance_cutoff`
- `metric::Distances.SemiMetric`: Distance metric (default: `Euclidean()`)

# Notes
When `distance_cutoff` is restrictive relative to the data, the candidate pool may
exhaust before `n_train` samples have been selected. The train cohort is then
returned smaller than requested and a `@warn` is emitted with
`_id = :datasplits_optisim_undershoot` and `_group = :datasplits`.

To silence this warning for a batch of splits, filter by id (e.g. with
`LoggingExtras.EarlyFilteredLogger`):

```julia
using Logging, LoggingExtras
silent = EarlyFilteredLogger(log -> log.id !== :datasplits_optisim_undershoot,
                             current_logger())
with_logger(silent) do
    # repeated partition(...) calls here emit no undershoot warnings
end
```

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding
  Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.

# Examples
```julia
res = partition(X, OptiSimSplit(; max_subsample_size=10); train=70, test=30)
X_train, X_test = splitdata(res, X)
```
"""
struct OptiSimSplit <: AbstractSplitStrategy
  max_subsample_size::Int
  distance_cutoff::Float64
  metric::Distances.SemiMetric
end

function OptiSimSplit(;
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
  OptiSimSplit(Int(max_subsample_size), Float64(distance_cutoff), metric)
end

consumes(::OptiSimSplit) = (:data,)
fallback_from_data(::OptiSimSplit) = ()

function _partition(
  X,
  s::OptiSimSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  N = numobs(X)
  D = distance_matrix(X, s.metric)
  selected_positions =
    optisim(D, n_train, s.max_subsample_size, s.distance_cutoff; rng = rng)
  train_pos = collect(selected_positions)
  test_pos = setdiff(1:N, train_pos)
  _warn_optisim_undershoot(length(train_pos), n_train, s.distance_cutoff)
  return TrainTestSplit(train_pos, test_pos)
end

function _warn_optisim_undershoot(n_selected, n_requested, distance_cutoff)
  n_selected < n_requested || return
  @warn "OptiSim: selected $n_selected/$n_requested training samples; \
distance_cutoff=$distance_cutoff exhausted the candidate pool. \
Lower `distance_cutoff`, reduce `train`, or silence this warning by its \
`_id` (see `?OptiSimSplit`)." _id = :datasplits_optisim_undershoot _group = :datasplits
end

function optisim(
  D::AbstractMatrix,
  selected_samples::Int = 10,
  max_subsample_size::Int = 0,
  distance_cutoff::Float64 = 0.35;
  rng = Random.default_rng(),
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
