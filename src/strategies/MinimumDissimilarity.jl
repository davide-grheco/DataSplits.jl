"""
    MinimumDissimilaritySplit(frac; distance_cutoff=0.35, metric=Euclidean())

Greedy dissimilarity selection (Clark 1997). Alias for `OptiSimSplit` with
`max_subsample_size = 1` (considers only one candidate per iteration).

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding
  Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.

# Examples
```julia
res = partition(X, MinimumDissimilaritySplit(0.7))
X_train, X_test = splitdata(res, X)
```
"""
function MinimumDissimilaritySplit(frac::Real; distance_cutoff = 0.35, metric = Euclidean())
  OptiSimSplit(
    frac;
    max_subsample_size = 1,
    distance_cutoff = distance_cutoff,
    metric = metric,
  )
end

"""
    LazyMinimumDissimilaritySplit <: AbstractSplitStrategy

Lazy (on-the-fly distances) variant of `MinimumDissimilaritySplit`.

# Examples
```julia
res = partition(X, LazyMinimumDissimilaritySplit(0.7))
X_train, X_test = splitdata(res, X)
```
"""
struct LazyMinimumDissimilaritySplit <: AbstractSplitStrategy
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

consumes(::LazyMinimumDissimilaritySplit) = (:data,)
fallback_from_data(::LazyMinimumDissimilaritySplit) = ()

function _partition(X, s::LazyMinimumDissimilaritySplit; rng = Random.GLOBAL_RNG, kwargs...)
  _partition(
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
