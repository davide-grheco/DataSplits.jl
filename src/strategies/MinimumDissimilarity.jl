struct MinimumDissimilaritySplit <: SplitStrategy
  frac::ValidFraction{T}
  distance_cutoff::Float64
  metric::PreMetric
end

"""
    MinimumDissimilaritySplit(frac; distance_cutoff=0.35, metric=Euclidean())

Alias for `OptiSimSplit` with `max_subsample_size = 1`.

Implements a **greedy dissimilarity selection** strategy as described by Clark (1997),
where, at each iteration, only one candidate is considered for addition to the training
set. The dimension of the training set may vary depending on the distance cutoff selected.

Useful when speed is essential and dataset size is large, but selection diversity
is still desired.

Reference:
> Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.
"""
function MinimumDissimilaritySplit(frac::Real; distance_cutoff = 0.35, metric = Euclidean())
  MinimumDissimilaritySplit(ValidFraction(frac), distance_cutoff, metric)
end

function _split(X, s::MinimumDissimilaritySplit; rng = Random.GLOBAL_RNG)
  opti = OptiSimSplit(
    s.frac;
    selected_samples = s.selected_samples,
    max_subsample_size = 1,
    distance_cutoff = s.distance_cutoff,
    metric = s.metric,
  )
  return _split(X, opti; rng)
end
