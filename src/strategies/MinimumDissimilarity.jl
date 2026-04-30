"""
    MinimumDissimilaritySplit(; distance_cutoff=0.35, metric=Euclidean())

Greedy dissimilarity selection (Clark 1997). Alias for `OptiSimSplit` with
`max_subsample_size = 1` (considers only one candidate per iteration).

# References
- Clark, R. D. (1997). OptiSim: An Extended Dissimilarity Selection Method for Finding
  Diverse Representative Subsets. *J. Chem. Inf. Comput. Sci.*, 37(6), 1181–1188.

# Examples
```julia
res = partition(X, MinimumDissimilaritySplit(); train=70, test=30)
X_train, X_test = splitdata(res, X)
```
"""
function MinimumDissimilaritySplit(; distance_cutoff = 0.35, metric = Euclidean())
  OptiSimSplit(; max_subsample_size = 1, distance_cutoff = distance_cutoff, metric = metric)
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
struct LazyMinimumDissimilaritySplit <: AbstractSplitStrategy
  distance_cutoff::Float64
  metric::Distances.SemiMetric
end

function LazyMinimumDissimilaritySplit(; distance_cutoff = 0.35, metric = Euclidean())
  LazyMinimumDissimilaritySplit(distance_cutoff, metric)
end

consumes(::LazyMinimumDissimilaritySplit) = (:data,)
fallback_from_data(::LazyMinimumDissimilaritySplit) = ()

function _partition(
  X,
  s::LazyMinimumDissimilaritySplit;
  n_train,
  n_test,
  rng = Random.GLOBAL_RNG,
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
