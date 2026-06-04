"""
    MinimumDissimilaritySplit <: AbstractSplitStrategy

Greedy dissimilarity selection (Clark 1997). Equivalent to `OptiSimSplit` with
`max_subsample_size = 1` (considers only one candidate per iteration).

# Fields
- `distance_cutoff::Float64`: Similarity threshold (default: 0.35).
- `metric::Distances.SemiMetric`: Distance metric (default: `Euclidean()`).

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding
  Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.

# Examples
```julia
res = partition(X, MinimumDissimilaritySplit(); train=70, test=30)
X_train, X_test = splitdata(res, X)
```
"""
struct MinimumDissimilaritySplit <: AbstractSplitStrategy
  distance_cutoff::Float64
  metric::Distances.SemiMetric
end

function MinimumDissimilaritySplit(; distance_cutoff = 0.35, metric = Euclidean())
  distance_cutoff >= 0 || throw(
    SplitParameterError("`distance_cutoff` must be non-negative, got $distance_cutoff."),
  )
  MinimumDissimilaritySplit(Float64(distance_cutoff), metric)
end

consumes(::MinimumDissimilaritySplit) = (:data,)
fallback_from_data(::MinimumDissimilaritySplit) = ()

function _partition(
  X,
  s::MinimumDissimilaritySplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  _partition(
    X,
    OptiSimSplit(;
      max_subsample_size = 1,
      distance_cutoff = s.distance_cutoff,
      metric = s.metric,
    );
    n_train = n_train,
    n_test = n_test,
    rng = rng,
  )
end

"""
    LazyMinimumDissimilaritySplit <: AbstractSplitStrategy

Lazy (on-the-fly distances) variant of `MinimumDissimilaritySplit`.

# Examples
```julia
res = partition(X, LazyMinimumDissimilaritySplit(); train=70, test=30)
X_train, X_test = splitdata(res, X)
```
"""
struct LazyMinimumDissimilaritySplit{M<:Distances.SemiMetric} <: AbstractSplitStrategy
  distance_cutoff::Float64
  metric::M
end

function LazyMinimumDissimilaritySplit(; distance_cutoff = 0.35, metric = Euclidean())
  distance_cutoff >= 0 || throw(
    SplitParameterError("`distance_cutoff` must be non-negative, got $distance_cutoff."),
  )
  LazyMinimumDissimilaritySplit(distance_cutoff, metric)
end

consumes(::LazyMinimumDissimilaritySplit) = (:data,)
fallback_from_data(::LazyMinimumDissimilaritySplit) = ()

function _partition(
  X,
  s::LazyMinimumDissimilaritySplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  _partition(
    X,
    LazyOptiSimSplit(;
      max_subsample_size = 1,
      distance_cutoff = s.distance_cutoff,
      metric = s.metric,
    );
    n_train = n_train,
    n_test = n_test,
    rng = rng,
  )
end
