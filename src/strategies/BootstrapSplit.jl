"""
    BootstrapSplit(n_splits::Integer) <: AbstractCVStrategy

Bootstrap cross-validation. For each of `n_splits` iterations the
train cohort is drawn from `1:N` with replacement (so it has exactly
`N` indices, with duplicates), and the test cohort is the set of
**out-of-bag (OOB)** observations — the unique indices that were not
sampled. On average about `(1 - 1/e) ≈ 63.2%` of unique observations
land in train; the remaining `~36.8%` form the OOB test.

# Fields
- `n_splits::Int`: Number of bootstrap resamples (must be ≥ 1).

# Important notes

- **Train contains duplicates by design.** This is the defining
  property of bootstrap sampling — it lets the model see the variance
  introduced by resampling the empirical distribution. If you need
  unique-only train indices use [`ShuffleSplit`](@ref) instead.
- **Test (OOB) size varies fold-to-fold.** It is whatever the
  bootstrap left out; no caller-set size is honoured.
- Cohort sizes (`train`, `test`) are not accepted: train is always
  `N` (with replacement), test is the OOB.
- Almost surely non-empty test for `N ≥ 2` (probability of all `N`
  indices being drawn at least once falls off as `N! / N^N`).

# Examples
```julia
# 50 bootstrap resamples.
cvs = partition(X, BootstrapSplit(50); rng = MersenneTwister(42))

for (X_train, X_test) in splitview(cvs, X)
    fit!(model, X_train)        # X_train has N obs, with duplicates
    evaluate(model, X_test)     # X_test is the OOB
end
```
"""
struct BootstrapSplit <: AbstractCVStrategy
  n_splits::Int
end

BootstrapSplit(n_splits::Integer) = BootstrapSplit(Int(n_splits))

consumes(::BootstrapSplit) = ()
fallback_from_data(::BootstrapSplit) = ()

function _partition(data, alg::BootstrapSplit; rng = Random.default_rng(), kwargs...)
  alg.n_splits >= 1 || throw(
    SplitParameterError(
      "BootstrapSplit requires n_splits ≥ 1, got n_splits=$(alg.n_splits).",
    ),
  )

  N = numobs(data)

  result = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.n_splits)
  for i = 1:alg.n_splits
    train_idx = rand(rng, 1:N, N)
    in_bag = falses(N)
    for j in train_idx
      in_bag[j] = true
    end
    test_idx = findall(!, in_bag)
    !isempty(test_idx) || throw(
      SplitParameterError(
        "BootstrapSplit: resample $i drew every observation; out-of-bag test cohort is empty. Try a different rng or a larger N.",
      ),
    )
    result[i] = TrainTestSplit(train_idx, test_idx)
  end
  return CrossValidationSplit(result)
end
