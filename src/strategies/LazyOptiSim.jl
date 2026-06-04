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
struct LazyOptiSimSplit{M<:Distances.SemiMetric} <: AbstractSplitStrategy
  max_subsample_size::Int
  distance_cutoff::Float64
  metric::M
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
  LazyOptiSimSplit(max_subsample_size, Float64(distance_cutoff), metric)
end

consumes(::LazyOptiSimSplit) = (:data,)
fallback_from_data(::LazyOptiSimSplit) = ()

function _lazy_update_dists!(min_dist, X, ref, candidates, metric)
  obs = _obs(X, ref)
  @inbounds for i in candidates
    d = Distances.evaluate(metric, _obs(X, i), obs)
    if d < min_dist[i]
      min_dist[i] = d
    end
  end
end

function _partition(
  X,
  s::LazyOptiSimSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  N = numobs(X)
  K = s.max_subsample_size
  metric, cutoff = _optisim_metric(s.metric, s.distance_cutoff)

  candidates = collect(1:N)
  # min_dist[i] = minimum distance from i to any selected point.
  # All candidates are pruned to min_dist >= cutoff after each selection,
  # so subsample building never needs to filter.
  min_dist = fill(Inf, N)

  idx = rand(rng, 1:N)
  first_sel = candidates[idx]
  candidates[idx] = candidates[end]
  pop!(candidates)
  selected = [first_sel]
  sizehint!(selected, n_train)

  obs_ref = _obs(X, first_sel)
  @inbounds for i in candidates
    min_dist[i] = Distances.evaluate(metric, _obs(X, i), obs_ref)
  end
  _prune_similar!(candidates, min_dist, cutoff)

  while length(selected) < n_train && !isempty(candidates)
    nc = length(candidates)
    use_all = K == 0 || K >= nc
    k = use_all ? nc : K

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

    _lazy_update_dists!(min_dist, X, best, candidates, metric)
    _prune_similar!(candidates, min_dist, cutoff)
  end

  train_pos = selected
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
