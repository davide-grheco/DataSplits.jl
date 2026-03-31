"""
    MaximumDissimilaritySplit(frac; distance_cutoff=0.35, metric=Euclidean())

Full OptiSim strategy (Clark 1997). Alias for `OptiSimSplit` with
`max_subsample_size = N` — considers all remaining candidates each iteration.

# Notes
- Greedily includes outliers; remove them before splitting if not desired.

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding
  Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.

# Examples
```julia
res = partition(X, MaximumDissimilaritySplit(0.7))
X_train, X_test = splitdata(res, X)
```
"""
struct MaximumDissimilaritySplit <: AbstractSplitStrategy
  frac::ValidFraction
  distance_cutoff::Float64
  metric::PreMetric
end

function MaximumDissimilaritySplit(frac::Real; distance_cutoff = 0.35, metric = Euclidean())
  MaximumDissimilaritySplit(ValidFraction(frac), distance_cutoff, metric)
end

consumes(::MaximumDissimilaritySplit) = (:data,)
fallback_from_data(::MaximumDissimilaritySplit) = ()

function _partition(X, s::MaximumDissimilaritySplit; rng = Random.GLOBAL_RNG, kwargs...)
  N = numobs(X)
  _partition(
    X,
    OptiSimSplit(
      s.frac;
      max_subsample_size = N,
      distance_cutoff = s.distance_cutoff,
      metric = s.metric,
    );
    rng,
  )
end

"""
    LazyMaximumDissimilaritySplit <: AbstractSplitStrategy

Lazy (on-the-fly distances) variant of `MaximumDissimilaritySplit`.

# Examples
```julia
res = partition(X, LazyMaximumDissimilaritySplit(0.7))
X_train, X_test = splitdata(res, X)
```
"""
struct LazyMaximumDissimilaritySplit <: AbstractSplitStrategy
  frac::ValidFraction
  distance_cutoff::Float64
  metric::Distances.SemiMetric
end

function LazyMaximumDissimilaritySplit(
  frac::Real;
  distance_cutoff = 0.35,
  metric = Euclidean(),
)
  LazyMaximumDissimilaritySplit(ValidFraction(frac), distance_cutoff, metric)
end

consumes(::LazyMaximumDissimilaritySplit) = (:data,)
fallback_from_data(::LazyMaximumDissimilaritySplit) = ()

function _partition(X, s::LazyMaximumDissimilaritySplit; rng = Random.GLOBAL_RNG, kwargs...)
  N = numobs(X)
  _partition(
    X,
    LazyOptiSimSplit(
      s.frac;
      max_subsample_size = N,
      distance_cutoff = s.distance_cutoff,
      metric = s.metric,
    );
    rng = rng,
  )
end
