"""
    RepeatedStratifiedKFold(k::Integer; n_repeats::Integer=10, bins::Integer=10) <: AbstractCVStrategy

Repeated stratified k-fold cross-validation. Runs
[`StratifiedKFold`](@ref) `n_repeats` times with a fresh random
permutation each repeat, producing `k * n_repeats` folds in total.
Mirrors scikit-learn's `RepeatedStratifiedKFold`.

The `target=` keyword is required (or, by fallback, `data` itself
plays that role); see `StratifiedKFold` for the stratification rule
(unique values for discrete targets, quantile bins for floats).

# Fields
- `k::Int`: Number of folds per repeat (must be ≥ 2 and ≤ N).
- `n_repeats::Int`: Number of independent stratified K-fold partitions
  (must be ≥ 1).
- `bins::Int`: Number of quantile bins for floating-point targets
  (must be ≥ 2; default `10`).

# Examples
```julia
# Classification.
cvs = partition(X, RepeatedStratifiedKFold(5; n_repeats = 10);
                target = labels, rng = MersenneTwister(42))

# Regression with 4 quantile bins.
cvs = partition(X, RepeatedStratifiedKFold(5; n_repeats = 10, bins = 4);
                target = y_continuous)
```
"""
struct RepeatedStratifiedKFold <: AbstractCVStrategy
  k::Int
  n_repeats::Int
  bins::Int
end

RepeatedStratifiedKFold(k::Integer; n_repeats::Integer = 10, bins::Integer = 10) =
  RepeatedStratifiedKFold(Int(k), Int(n_repeats), Int(bins))

consumes(::RepeatedStratifiedKFold) = (:target,)
fallback_from_data(::RepeatedStratifiedKFold) = (:target,)

function _partition(
  data,
  alg::RepeatedStratifiedKFold;
  target,
  rng = Random.default_rng(),
  kwargs...,
)
  alg.n_repeats >= 1 || throw(
    SplitParameterError(
      "RepeatedStratifiedKFold requires n_repeats ≥ 1, got n_repeats=$(alg.n_repeats).",
    ),
  )

  inner = StratifiedKFold(alg.k; bins = alg.bins, shuffle = true)
  all_folds = TrainTestSplit{Vector{Int}}[]
  for _ = 1:alg.n_repeats
    sub = _partition(data, inner; target = target, rng = rng)
    append!(all_folds, folds(sub))
  end
  return CrossValidationSplit(all_folds)
end
