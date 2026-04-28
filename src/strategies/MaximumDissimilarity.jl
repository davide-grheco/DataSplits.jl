"""
    MaximumDissimilaritySplit(; distance_cutoff=0.35, metric=Euclidean())

Full OptiSim strategy (Clark 1997). Alias for `OptiSimSplit` with
`max_subsample_size = N` — considers all remaining candidates each iteration.

# Notes
- Greedily includes outliers; remove them before splitting if not desired.

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding
  Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.

# Examples
```julia
res = partition(X, MaximumDissimilaritySplit(); train=70, test=30)
X_train, X_test = splitdata(res, X)
```
"""
struct MaximumDissimilaritySplit <: AbstractSplitStrategy
  distance_cutoff::Float64
  metric::PreMetric
end

function MaximumDissimilaritySplit(; distance_cutoff = 0.35, metric = Euclidean())
  MaximumDissimilaritySplit(distance_cutoff, metric)
end

consumes(::MaximumDissimilaritySplit) = (:data,)
fallback_from_data(::MaximumDissimilaritySplit) = ()

function _partition(
  X,
  s::MaximumDissimilaritySplit;
  n_train,
  n_test,
  rng = Random.GLOBAL_RNG,
  kwargs...,
)
  N = numobs(X)
  _partition(
    X,
    OptiSimSplit(;
      max_subsample_size = N,
      distance_cutoff = s.distance_cutoff,
      metric = s.metric,
    );
    n_train = n_train,
    n_test = n_test,
    rng = rng,
  )
end

"""
    LazyMaximumDissimilaritySplit <: AbstractSplitStrategy

Lazy (on-the-fly distances) variant of `MaximumDissimilaritySplit`.

# Examples
```julia
res = partition(X, LazyMaximumDissimilaritySplit(); train=70, test=30)
X_train, X_test = splitdata(res, X)
```
"""
struct LazyMaximumDissimilaritySplit <: AbstractSplitStrategy
  distance_cutoff::Float64
  metric::Distances.SemiMetric
end

function LazyMaximumDissimilaritySplit(; distance_cutoff = 0.35, metric = Euclidean())
  LazyMaximumDissimilaritySplit(distance_cutoff, metric)
end

consumes(::LazyMaximumDissimilaritySplit) = (:data,)
fallback_from_data(::LazyMaximumDissimilaritySplit) = ()

function _partition(
  X,
  s::LazyMaximumDissimilaritySplit;
  n_train,
  n_test,
  rng = Random.GLOBAL_RNG,
  kwargs...,
)
  N = numobs(X)
  _partition(
    X,
    LazyOptiSimSplit(;
      max_subsample_size = N,
      distance_cutoff = s.distance_cutoff,
      metric = s.metric,
    );
    n_train = n_train,
    n_test = n_test,
    rng = rng,
  )
end
