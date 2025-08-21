struct MaximumDissimilaritySplit <: SplitStrategy
  frac::ValidFraction
  distance_cutoff::Float64
  metric::PreMetric
end

"""
    MaximumDissimilaritySplit(frac; distance_cutoff=0.35, metric=Euclidean())

Full OptiSim strategy for train/test splitting (Clark 1997).

Alias for `OptiSimSplit` with `max_subsample_size = N`, where `N` is the number of samples.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `distance_cutoff::Float64`: Dissimilarity threshold (default: 0.35)
- `metric::PreMetric`: Distance metric (default: Euclidean())

# Notes
- At each iteration, all remaining candidates are considered; the one that maximizes the minimum dissimilarity to the selected set is chosen.
- Greedily includes outliers; remove outliers before splitting if not desired.
- Does not discard initially selected samples.

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.

# Examples
```julia
splitter = MaximumDissimilaritySplit(0.7)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```
"""

function MaximumDissimilaritySplit(frac::Real; distance_cutoff = 0.35, metric = Euclidean())
  MaximumDissimilaritySplit(ValidFraction(frac), distance_cutoff, metric)
end

function _split(X, s::MaximumDissimilaritySplit; rng = Random.GLOBAL_RNG)
  N = numobs(X)
  opti = OptiSimSplit(
    s.frac;
    max_subsample_size = N,
    distance_cutoff = s.distance_cutoff,
    metric = s.metric,
  )
  return _split(X, opti; rng)
end
