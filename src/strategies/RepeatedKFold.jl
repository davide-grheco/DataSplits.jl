"""
    RepeatedKFold(k::Integer; n_repeats::Integer=10) <: AbstractCVStrategy

Repeated k-fold cross-validation. Runs [`KFold`](@ref) `n_repeats`
times with a fresh random permutation each repeat, producing
`k * n_repeats` folds in total. Mirrors scikit-learn's `RepeatedKFold`.

Each repeat is a full k-fold partition of the data; across repeats the
fold assignments are independent random permutations. Use the same
`rng` (with a fixed seed) to reproduce the full set of folds.

# Fields
- `k::Int`: Number of folds per repeat (must be ≥ 2 and ≤ N).
- `n_repeats::Int`: Number of independent K-fold partitions (must be ≥ 1).

# Examples
```julia
# 50 folds total (5 folds × 10 repeats).
cvs = partition(X, RepeatedKFold(5; n_repeats = 10);
                rng = MersenneTwister(42))
```
"""
struct RepeatedKFold <: AbstractCVStrategy
  k::Int
  n_repeats::Int
end

RepeatedKFold(k::Integer; n_repeats::Integer = 10) =
  RepeatedKFold(Int(k), Int(n_repeats))

consumes(::RepeatedKFold) = ()
fallback_from_data(::RepeatedKFold) = ()

function _partition(data, alg::RepeatedKFold; rng = Random.default_rng(), kwargs...)
  alg.n_repeats >= 1 || throw(
    SplitParameterError(
      "RepeatedKFold requires n_repeats ≥ 1, got n_repeats=$(alg.n_repeats).",
    ),
  )

  inner = KFold(alg.k, true)
  all_folds = TrainTestSplit{Vector{Int}}[]
  for _ = 1:alg.n_repeats
    sub = _partition(data, inner; rng = rng)
    append!(all_folds, folds(sub))
  end
  return CrossValidationSplit(all_folds)
end
