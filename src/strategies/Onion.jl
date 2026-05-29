"""
    OnionSplit <: AbstractSplitStrategy

Onion algorithm for train/test splitting.

Selects representative calibration and test samples by iteratively peeling
onion-like layers based on Euclidean distance from the centroid of `X`.
The outer layers (farthest from the centroid) go to the training set; the
next-outermost to the test set; remaining samples are assigned randomly in
the requested proportion.

This is the X-only counterpart of [`XYOnionSplit`](@ref): target values are not
used.  The split sizes are approximate — rounding in each layer means the final
train/test counts may differ slightly from the requested percentages.

# Fields
- `n_layers::Int`: Number of onion layers (default: `3`)
- `metric_X::Union{Nothing,Distances.SemiMetric}`: Distance metric for `X`.
  `nothing` computes Mahalanobis distance via eigendecomposition of the data
  covariance at split time.  `Euclidean()` (default) uses standard Euclidean
  distance.

# Examples
```julia
res = partition(X, OnionSplit(); train = 70, test = 30)
res = partition(X, OnionSplit(; metric_X = nothing); train = 70, test = 30)
X_train, X_test = splitdata(res, X)
```

# References
Gallagher, N.B.; O'Sullivan, D. *Selection of Representative Learning and Test
Sets Using the Onion Method.* Eigenvector Research Technical Report (2022).
https://eigenvector.com/wp-content/uploads/2022/10/Onion_SampleSelection.pdf

# See also
[`XYOnionSplit`](@ref) — joint X–y variant of this algorithm.
"""
struct OnionSplit <: AbstractSplitStrategy
  n_layers::Int
  metric_X::Union{Nothing,Distances.SemiMetric}
end

OnionSplit(; n_layers::Int = 3, metric_X = Euclidean()) = OnionSplit(n_layers, metric_X)

consumes(::OnionSplit) = (:data,)
fallback_from_data(::OnionSplit) = ()

function _partition(
  X,
  s::OnionSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  Xf = float.(X)
  Xw = s.metric_X === nothing ? _xyonion_whiten(Xf) : Xf
  train_idx, test_idx = _onion_partition!(Xw, nothing, n_train, n_test, s.n_layers, rng)
  return TrainTestSplit(train_idx, test_idx)
end

function _partition(
  X::AbstractVector{<:AbstractVector},
  s::OnionSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  _partition(stack(X), s; n_train, n_test, rng)
end
