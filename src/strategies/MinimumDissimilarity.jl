"""
    MinimumDissimilaritySplit(frac; distance_cutoff=0.35, metric=Euclidean())

Greedy dissimilarity selection strategy for train/test splitting (Clark 1997).

Alias for `OptiSimSplit` with `max_subsample_size = 1`.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)
- `distance_cutoff::Float64`: Dissimilarity threshold (default: 0.35)
- `metric::PreMetric`: Distance metric (default: Euclidean())

# Notes
- At each iteration, only one candidate is considered for addition to the training set.
- The training set size may vary depending on the distance cutoff.

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.

# Examples
```julia
splitter = MinimumDissimilaritySplit(0.7)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```
"""
function MinimumDissimilaritySplit(frac::Real; distance_cutoff = 0.35, metric = Euclidean())
  OptiSimSplit(
    frac,
    max_subsample_size = 1,
    distance_cutoff = distance_cutoff,
    metric = metric,
  )
end



struct LazyMinimumDissimilaritySplit <: SplitStrategy
  frac::ValidFraction
  distance_cutoff::Float64
  metric::Distances.SemiMetric
end

function LazyMinimumDissimilaritySplit(
  frac::Real;
  distance_cutoff = 0.35,
  metric = Euclidean(),
)
  LazyMinimumDissimilaritySplit(ValidFraction(frac), distance_cutoff, metric)
end

function _split(X, s::LazyMinimumDissimilaritySplit; rng = Random.GLOBAL_RNG)
  return _split(
    X,
    LazyOptiSimSplit(
      s.frac;
      max_subsample_size = 1,
      distance_cutoff = s.distance_cutoff,
      metric = s.metric,
    );
    rng = rng,
  )
end
