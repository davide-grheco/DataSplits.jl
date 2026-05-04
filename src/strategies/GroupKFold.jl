"""
    GroupKFold(k::Integer; shuffle::Bool=false) <: AbstractCVStrategy

Group-aware k-fold cross-validation. Whole groups are assigned to a single
fold; no group ever appears in both the train and test cohort of the same
fold. Equivalent in spirit to scikit-learn's `GroupKFold`.

Groups are passed via the `groups=` keyword (or, by fallback, `data` itself
plays that role).

# Fields
- `k::Int`: Number of folds (must be ≥ 2 and ≤ number of unique groups).
- `shuffle::Bool`: When `true`, the order in which groups are considered
  for fold assignment is shuffled using the `rng` passed to `partition`,
  so different RNG seeds yield different fold compositions. When `false`
  (default), assignment is deterministic and reproducible without an `rng`.

# Notes
- Within each candidate group, the algorithm places it in the currently
  smallest fold, so observation counts across folds stay roughly balanced
  whether or not shuffling is enabled. Mirrors sklearn's `GroupKFold`.
- When `shuffle=false`, groups are processed in **descending order of size**
  (the classic deterministic balancing). When `shuffle=true`, they are
  processed in a randomly permuted order — folds remain balanced but no
  longer follow size order.

# Examples
```julia
# Deterministic (default).
cvs = partition(X, GroupKFold(5); groups = patient_ids)

# Shuffled — different seeds give different fold compositions.
cvs = partition(X, GroupKFold(5; shuffle = true);
                groups = patient_ids, rng = MersenneTwister(42))

for (X_train, X_test) in splitview(cvs, X)
    # train and evaluate
end

# Fallback: ids are simultaneously the data and the groups.
cvs = partition(patient_ids, GroupKFold(5))
```
"""
struct GroupKFold <: AbstractCVStrategy
  k::Int
  shuffle::Bool
end

GroupKFold(k::Integer; shuffle::Bool = false) = GroupKFold(Int(k), shuffle)

consumes(::GroupKFold) = (:groups,)
fallback_from_data(::GroupKFold) = (:groups,)

function _partition(data, alg::GroupKFold; groups, rng = Random.default_rng(), kwargs...)
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

  if alg.shuffle
    Random.shuffle!(rng, unique_groups)
  else
    sort!(unique_groups; by = g -> -length(group_to_indices[g]))
  end

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
