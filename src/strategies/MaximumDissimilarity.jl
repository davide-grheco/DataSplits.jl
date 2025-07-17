struct MaximumDissimilaritySplit <: SplitStrategy
  frac::ValidFraction
  distance_cutoff::Float64
  metric::PreMetric
end

"""
    MaximumDissimilaritySplit(frac; distance_cutoff=0.35, metric=Euclidean())

Alias for `OptiSimSplit` with `max_subsample_size = N`, where `N` is the number of
samples.

Implements the **full OptiSim strategy** as described by Clark (1997), considering all
remaining candidates at each iteration and selecting the one that maximizes the
minimum dissimilarity to the selected set.

Best suited when high-quality representative subsets are essential and computational
resources allow.

Note:

- This method greedily includes outliers, if that is not desired outliers removal should be performed prior to splitting.
- This does not discard initially selected samples to account for the initial random selection.

Reference:
> Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.
"""

function MaximumDissimilaritySplit(frac::Real; distance_cutoff = 0.35, metric = Euclidean())
  MaximumDissimilaritySplit(ValidFraction(frac), distance_cutoff, metric)
end

function _split(X, s::MaximumDissimilaritySplit; rng = Random.GLOBAL_RNG)
  N = length(sample_indices(X))
  opti = OptiSimSplit(
    s.frac;
    selected_samples = s.selected_samples,
    max_subsample_size = N,
    distance_cutoff = s.distance_cutoff,
    metric = s.metric,
  )
  return _split(X, opti; rng)
end
