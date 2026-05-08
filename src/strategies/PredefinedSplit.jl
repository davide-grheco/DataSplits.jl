"""
    PredefinedSplit(test_fold::AbstractVector{<:Integer}) <: AbstractCVStrategy

Cross-validation with caller-provided fold assignments. Each entry of
`test_fold` gives the fold ID in which the corresponding observation
serves as **test**. Negative values mean the observation is **never**
placed in the test cohort — it is part of every fold's training set.

Folds are produced in ascending order of fold ID. Mirrors
scikit-learn's `PredefinedSplit`, but driven by `partition` and
DataSplits' `CrossValidationSplit` result type.

# Fields
- `test_fold::Vector{Int}`: Length-`N` vector mapping each observation to
  the fold ID where it tests, or to a negative value for "always train".

# Notes
- `length(test_fold)` must equal `numobs(data)`.
- Fold IDs need not be contiguous; what matters is the set of distinct
  non-negative values. At least one non-negative ID must exist.

# Examples
```julia
# Three folds: obs 1-2 test in fold 0, obs 3-4 test in fold 1, obs 5-6
# test in fold 2.
test_fold = [0, 0, 1, 1, 2, 2]
cvs = partition(X, PredefinedSplit(test_fold))

# Hold-out style: obs 7-10 are reserved for training only across all folds.
test_fold = [0, 0, 0, 1, 1, 1, -1, -1, -1, -1]
cvs = partition(X, PredefinedSplit(test_fold))
```
"""
struct PredefinedSplit <: AbstractCVStrategy
  test_fold::Vector{Int}
end

PredefinedSplit(test_fold::AbstractVector{<:Integer}) =
  PredefinedSplit(collect(Int, test_fold))

consumes(::PredefinedSplit) = ()
fallback_from_data(::PredefinedSplit) = ()

function _partition(data, alg::PredefinedSplit; kwargs...)
  N = numobs(data)
  length(alg.test_fold) == N || throw(
    SplitInputError(
      "PredefinedSplit: `test_fold` length ($(length(alg.test_fold))) does not match number of observations ($N).",
    ),
  )

  fold_ids = sort!(unique(f for f in alg.test_fold if f >= 0))
  isempty(fold_ids) && throw(
    SplitParameterError(
      "PredefinedSplit requires at least one non-negative fold ID in `test_fold`; all entries were negative.",
    ),
  )

  folds = Vector{TrainTestSplit{Vector{Int}}}(undef, length(fold_ids))
  for (k, f) in enumerate(fold_ids)
    test_idx = findall(==(f), alg.test_fold)
    train_idx = findall(!=(f), alg.test_fold)
    # Observations marked with a negative fold are part of train for every fold.
    # `findall(!=(f))` already includes them since they never equal a non-negative `f`.
    folds[k] = TrainTestSplit(train_idx, test_idx)
  end
  return CrossValidationSplit(folds)
end
