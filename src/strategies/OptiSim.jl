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
  eff_metric, eff_cutoff = _optisim_metric(s.metric, s.distance_cutoff)
  D = distance_matrix(X, eff_metric)
  selected_positions = optisim(D, n_train, s.max_subsample_size, eff_cutoff; rng = rng)
  train_pos = collect(selected_positions)
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

  candidates = collect(1:N)
  # min_dist[i] = minimum distance from i to any selected point.
  # Maintained incrementally; all candidates are pruned to min_dist >= cutoff
  # after each selection, so subsample building never needs to filter.
  min_dist = fill(Inf, N)

  idx = rand(rng, 1:N)
  first_sel = candidates[idx]
  candidates[idx] = candidates[end]
  pop!(candidates)
  selected = [first_sel]
  sizehint!(selected, M)

  col_first = @view D[:, first_sel]
  @inbounds @simd for i = 1:N
    min_dist[i] = col_first[i]
  end
  _prune_similar!(candidates, min_dist, distance_cutoff)

  while length(selected) < M && !isempty(candidates)
    nc = length(candidates)
    use_all = K == 0 || K >= nc
    k = use_all ? nc : K

    # Find the maximin candidate: either scan all (use_all) or a random k-subset
    # via partial Fisher-Yates in-place on candidates. All candidates are
    # dissimilar by invariant, so no filtering is needed during the scan.
    best_idx = 1
    if use_all
      best_score = min_dist[candidates[1]]
      @inbounds for j = 2:nc
        sc = min_dist[candidates[j]]
        if sc > best_score
          best_score = sc
          best_idx = j
        end
      end
    else
      @inbounds for j = 1:k
        ri = rand(rng, j:nc)
        candidates[j], candidates[ri] = candidates[ri], candidates[j]
      end
      best_score = min_dist[candidates[1]]
      @inbounds for j = 2:k
        sc = min_dist[candidates[j]]
        if sc > best_score
          best_score = sc
          best_idx = j
        end
      end
    end

    best = candidates[best_idx]
    candidates[best_idx] = candidates[end]
    pop!(candidates)
    push!(selected, best)

    col_best = @view D[:, best]
    @inbounds @simd for i = 1:N
      min_dist[i] = min(min_dist[i], col_best[i])
    end
    _prune_similar!(candidates, min_dist, distance_cutoff)
  end

  return Set(selected)
end
