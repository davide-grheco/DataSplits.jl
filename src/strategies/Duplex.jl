using Distances

"""
    DuplexSplit <: AbstractSplitStrategy

Duplex algorithm for train/test splitting.

Unlike Kennard–Stone, which only ensures the *training* set covers the feature
space, Duplex builds both cohorts **simultaneously**: the training and test sets are
constructed in parallel via alternating maximin selection, so each independently
covers the data space.

Starts by assigning the globally most distant pair to train and test respectively,
then alternately adds to each cohort the unassigned sample that maximises the minimum
distance to that cohort's existing members.

# Fields
- `metric::Distances.SemiMetric`: Distance metric (default: `Euclidean()`)

# Notes
- Split sizes are exact: `n_train` and `n_test` are filled precisely.
- Precomputes the full N×N distance matrix (O(N²) memory). Use
  [`LazyDuplexSplit`](@ref) when that is prohibitive.

# Examples
```julia
res = partition(X, DuplexSplit(); train = 70, test = 30)
X_train, X_test = splitdata(res, X)

res = partition(X, DuplexSplit(Mahalanobis(cov(X; dims=2))); train = 0.8, test = 0.2)
```

# See also
[`LazyDuplexSplit`](@ref), [`KennardStoneSplit`](@ref)

# References
Snee, R. D. Validation of Regression Models: Methods and Examples.
*Technometrics* 1977, 19(4), 415–428. <https://doi.org/10.2307/1267881>.
"""
struct DuplexSplit <: AbstractSplitStrategy
  metric::Distances.SemiMetric
end

DuplexSplit() = DuplexSplit(Euclidean())

"""
    LazyDuplexSplit <: AbstractSplitStrategy

Memory-efficient Duplex algorithm. Computes distances on-the-fly (O(N) storage)
rather than precomputing the full N×N matrix.

See [`DuplexSplit`](@ref) for the algorithm description and usage examples.
"""
struct LazyDuplexSplit <: AbstractSplitStrategy
  metric::Distances.SemiMetric
end

LazyDuplexSplit() = LazyDuplexSplit(Euclidean())

consumes(::DuplexSplit) = (:data,)
fallback_from_data(::DuplexSplit) = ()

consumes(::LazyDuplexSplit) = (:data,)
fallback_from_data(::LazyDuplexSplit) = ()

# ---------------------------------------------------------------------------
# Core eager algorithm
# ---------------------------------------------------------------------------

"""
    duplex_from_distance_matrix(D, n_train, n_test) -> (train, test)

Duplex selection on a precomputed N×N distance matrix `D`.
Returns index vectors of length `n_train` and `n_test`.
"""
function duplex_from_distance_matrix(D::AbstractMatrix, n_train::Integer, n_test::Integer)
  N = size(D, 1)
  i, j = find_most_distant_pair(D)

  train = Vector{Int}(undef, n_train)
  test = Vector{Int}(undef, n_test)
  train[1] = i
  test[1] = j

  # min distance from each sample to the current train / test set
  min_dist_train = copy(D[:, i])
  min_dist_test = copy(D[:, j])
  # seeds are no longer available for selection
  min_dist_train[[i, j]] .= -Inf
  min_dist_test[[i, j]] .= -Inf

  n_tr = 1
  n_te = 1

  while n_tr < n_train || n_te < n_test
    if n_tr < n_train
      k = argmax(min_dist_train)
      n_tr += 1
      train[n_tr] = k
      col_k = @view D[:, k]
      @inbounds @simd for m = 1:N
        min_dist_train[m] = min(min_dist_train[m], col_k[m])
      end
      min_dist_train[k] = -Inf
      min_dist_test[k] = -Inf  # cross-invalidate
    end
    if n_te < n_test
      l = argmax(min_dist_test)
      n_te += 1
      test[n_te] = l
      col_l = @view D[:, l]
      @inbounds @simd for m = 1:N
        min_dist_test[m] = min(min_dist_test[m], col_l[m])
      end
      min_dist_test[l] = -Inf
      min_dist_train[l] = -Inf  # cross-invalidate
    end
  end

  return train, test
end

# ---------------------------------------------------------------------------
# _partition methods
# ---------------------------------------------------------------------------

function _partition(
  X,
  s::DuplexSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  D = distance_matrix(X, _squared_metric(s.metric))
  train, test = duplex_from_distance_matrix(D, n_train, n_test)
  return TrainTestSplit(train, test)
end

function _partition(
  X::AbstractVector{<:AbstractVector},
  s::DuplexSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  _partition(stack(X), s; n_train, n_test, rng)
end

function _partition(
  data,
  s::LazyDuplexSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  N = numobs(data)
  metric = _squared_metric(s.metric)
  i, j = find_most_distant_pair(data, metric)

  train = Vector{Int}(undef, n_train)
  test = Vector{Int}(undef, n_test)
  train[1] = i
  test[1] = j

  selected = falses(N)
  selected[i] = selected[j] = true

  xi = _obs(data, i)
  xj = _obs(data, j)

  # Initialise min-distance vectors from the seed observations
  min_dist_train = fill(-Inf, N)
  min_dist_test = fill(-Inf, N)
  @inbounds for k = 1:N
    if !selected[k]
      xk = _obs(data, k)
      min_dist_train[k] = Distances.evaluate(metric, xk, xi)
      min_dist_test[k] = Distances.evaluate(metric, xk, xj)
    end
  end

  n_tr = 1
  n_te = 1

  while n_tr < n_train || n_te < n_test
    if n_tr < n_train
      k = argmax(min_dist_train)
      n_tr += 1
      train[n_tr] = k
      selected[k] = true
      min_dist_train[k] = -Inf
      min_dist_test[k] = -Inf
      xk = _obs(data, k)
      @inbounds for m = 1:N
        if !selected[m]
          d = Distances.evaluate(metric, _obs(data, m), xk)
          if d < min_dist_train[m]
            min_dist_train[m] = d
          end
        end
      end
    end
    if n_te < n_test
      l = argmax(min_dist_test)
      n_te += 1
      test[n_te] = l
      selected[l] = true
      min_dist_train[l] = -Inf
      min_dist_test[l] = -Inf
      xl = _obs(data, l)
      @inbounds for m = 1:N
        if !selected[m]
          d = Distances.evaluate(metric, _obs(data, m), xl)
          if d < min_dist_test[m]
            min_dist_test[m] = d
          end
        end
      end
    end
  end

  return TrainTestSplit(train, test)
end
