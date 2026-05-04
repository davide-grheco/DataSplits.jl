"""
    GroupKFold(k::Integer) <: AbstractCVStrategy

Group-aware k-fold cross-validation. Whole groups are assigned to a single
fold; no group ever appears in both the train and test cohort of the same
fold. Equivalent in spirit to scikit-learn's `GroupKFold`.

Groups are passed via the `groups=` keyword (or, by fallback, `data` itself
plays that role).

# Fields
- `k::Int`: Number of folds (must be ≥ 2 and ≤ number of unique groups).

# Notes
- Fold assignment is deterministic: groups are sorted by size (descending)
  and each is placed in the currently smallest fold. This yields folds of
  roughly equal observation count, like sklearn's `GroupKFold`.
- The `rng` keyword is accepted for interface uniformity but unused; results
  are reproducible without it.

# Examples
```julia
cvs = partition(X, GroupKFold(5); groups = patient_ids)

for (X_train, X_test) in splitview(cvs, X)
    # train and evaluate
end

# Fallback: ids are simultaneously the data and the groups.
cvs = partition(patient_ids, GroupKFold(5))
```
"""
struct GroupKFold <: AbstractCVStrategy
  k::Int
end

consumes(::GroupKFold) = (:groups,)
fallback_from_data(::GroupKFold) = (:groups,)

function _partition(data, alg::GroupKFold; groups, kwargs...)
  alg.k >= 2 || throw(SplitParameterError("GroupKFold requires k ≥ 2, got k=$(alg.k)."))

  N = numobs(data)
  # Anticipates issue #22 (length validation belongs in `partition`).
  length(groups) == N || throw(
    SplitInputError(
      "`groups` length ($(length(groups))) does not match number of observations ($N).",
    ),
  )

  group_to_indices = Dict{eltype(groups),Vector{Int}}()
  for (i, g) in enumerate(groups)
    push!(get!(group_to_indices, g, Int[]), i)
  end
  unique_groups = collect(keys(group_to_indices))
  n_groups = length(unique_groups)
  alg.k <= n_groups || throw(
    SplitParameterError(
      "GroupKFold(k=$(alg.k)) requires at least k unique groups; got $n_groups.",
    ),
  )

  sort!(unique_groups; by = g -> -length(group_to_indices[g]))

  fold_of = Dict{eltype(unique_groups),Int}()
  fold_counts = zeros(Int, alg.k)
  for g in unique_groups
    f = argmin(fold_counts)
    fold_of[g] = f
    fold_counts[f] += length(group_to_indices[g])
  end

  fold_test = [Int[] for _ = 1:alg.k]
  for i = 1:N
    push!(fold_test[fold_of[groups[i]]], i)
  end

  folds = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.k)
  for f = 1:alg.k
    test_set = Set(fold_test[f])
    train_idx = [i for i = 1:N if !(i in test_set)]
    folds[f] = TrainTestSplit(train_idx, fold_test[f])
  end
  return CrossValidationSplit(folds)
end
