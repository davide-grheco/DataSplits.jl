using Combinatorics: combinations

"""
    LeavePOut(p::Integer) <: AbstractCVStrategy

Exhaustive cross-validation that uses every possible combination of `p`
observations as the test set. Produces `binomial(n, p)` folds, where `n`
is the number of observations.

# Fields
- `p::Int`: Number of observations in each test fold (must be ≥ 1 and < n).

# Notes
- The number of folds grows as `binomial(n, p)`, which becomes very large
  quickly. Use only for small datasets or small values of `p`.
- For `p = 1`, prefer [`LeaveOneOut`](@ref) as a convenience alias.

# Examples
```julia
cvs = partition(X, LeavePOut(2))

for (X_train, X_test) in splitview(cvs, X)
    # train and evaluate
end
```
"""
struct LeavePOut <: AbstractCVStrategy
  p::Int
end

LeavePOut(p::Integer) = LeavePOut(Int(p))

consumes(::LeavePOut) = ()
fallback_from_data(::LeavePOut) = ()

function _partition(data, alg::LeavePOut; kwargs...)
  alg.p >= 1 || throw(SplitParameterError("LeavePOut requires p ≥ 1, got p=$(alg.p)."))

  N = numobs(data)
  alg.p < N || throw(
    SplitParameterError("LeavePOut(p=$(alg.p)) requires p < n observations; got n=$N."),
  )

  all_indices = collect(1:N)
  result = map(combinations(all_indices, alg.p)) do test_idx
    TrainTestSplit(setdiff(all_indices, test_idx), collect(test_idx))
  end

  return CrossValidationSplit(result)
end

# ---------------------------------------------------------------------------

"""
    LeaveOneOut() <: AbstractCVStrategy

Cross-validation where each single observation takes a turn as the test
set. Produces `n` folds (one per observation). Equivalent to `LeavePOut(1)`.

# Examples
```julia
cvs = partition(X, LeaveOneOut())

for (X_train, X_test) in splitview(cvs, X)
    # train and evaluate
end
```
"""
struct LeaveOneOut <: AbstractCVStrategy end

consumes(::LeaveOneOut) = ()
fallback_from_data(::LeaveOneOut) = ()

_partition(data, ::LeaveOneOut; kwargs...) = _partition(data, LeavePOut(1); kwargs...)
