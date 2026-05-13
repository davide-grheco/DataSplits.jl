"""
    StratifiedGroupKFold(k::Integer; bins::Integer=10, shuffle::Bool=false) <: AbstractCVStrategy

Group-aware **and** stratified k-fold cross-validation. No group spans
two folds (like [`GroupKFold`](@ref)), and per-fold class proportions
are kept close to the global distribution (like
[`StratifiedKFold`](@ref)). Mirrors scikit-learn's
`StratifiedGroupKFold`.

Both `target=` and `groups=` keywords are required; neither falls
back to `data` (no sensible default — they describe orthogonal
properties of the same observations).

# Fields
- `k::Int`: Number of folds (must be ≥ 2 and ≤ number of unique groups).
- `bins::Int`: Number of quantile bins used when `target` is
  floating-point (must be ≥ 2; default `10`). Ignored for non-float
  targets, which are treated as discrete classes.
- `shuffle::Bool`: When `true`, the order in which groups are
  considered for fold assignment is shuffled using the `rng` passed
  to `partition`. When `false` (default), groups are processed in
  descending order of class-count vector norm — sklearn's
  deterministic balancing.

# Algorithm

For each group, compute its per-class count vector. Process groups
one by one (largest first by default). For each group, assign it to
the fold that **minimises the variance of per-class proportions
across folds** after the assignment — each class's fold counts are
normalised by its global total so rare and abundant classes
contribute on the same scale. This is the same greedy heuristic
used by sklearn (`y_counts_per_fold / y_distr`).

# Notes
- Each class needs at least `k` members **and** must appear in at
  least `k` distinct groups; otherwise even the best assignment
  cannot place that class in every fold's training cohort. The
  algorithm does not raise on this — sklearn does not either — but
  fold class coverage may be uneven for very rare classes.

# Examples
```julia
# Classification with patient-level groups.
cvs = partition(X, StratifiedGroupKFold(5);
                target = labels, groups = patient_ids)

# Regression with quantile bins.
cvs = partition(X, StratifiedGroupKFold(5; bins = 4);
                target = y_continuous, groups = batch_ids)

# Shuffled ordering — different seeds give different fold compositions.
cvs = partition(X, StratifiedGroupKFold(5; shuffle = true);
                target = labels, groups = patient_ids,
                rng = MersenneTwister(42))
```
"""
struct StratifiedGroupKFold <: AbstractCVStrategy
  k::Int
  bins::Int
  shuffle::Bool
end

StratifiedGroupKFold(k::Integer; bins::Integer = 10, shuffle::Bool = false) =
  StratifiedGroupKFold(Int(k), Int(bins), shuffle)

consumes(::StratifiedGroupKFold) = (:target, :groups)
fallback_from_data(::StratifiedGroupKFold) = ()

function _partition(
  data,
  alg::StratifiedGroupKFold;
  target,
  groups,
  rng = Random.default_rng(),
  kwargs...,
)
  alg.k >= 2 || throw(
    SplitParameterError("StratifiedGroupKFold requires k ≥ 2, got k=$(alg.k)."),
  )
  alg.bins >= 2 || throw(
    SplitParameterError("StratifiedGroupKFold requires bins ≥ 2, got bins=$(alg.bins)."),
  )

  N = numobs(data)
  classes = _stratification_classes(target, alg.bins)

  # Distinct classes (column index of the per-fold count matrix).
  unique_classes = unique(classes)
  C = length(unique_classes)
  class_index = Dict(c => i for (i, c) in enumerate(unique_classes))

  # Per-group: indices and class counts.
  group_to_indices = Dict{eltype(groups),Vector{Int}}()
  for (i, g) in enumerate(groups)
    push!(get!(group_to_indices, g, Int[]), i)
  end
  unique_groups = collect(keys(group_to_indices))
  n_groups = length(unique_groups)
  alg.k <= n_groups || throw(
    SplitParameterError(
      "StratifiedGroupKFold(k=$(alg.k)) requires at least k unique groups; got $n_groups.",
    ),
  )

  group_class_counts = Dict{eltype(unique_groups),Vector{Int}}()
  for g in unique_groups
    cnt = zeros(Int, C)
    for i in group_to_indices[g]
      cnt[class_index[classes[i]]] += 1
    end
    group_class_counts[g] = cnt
  end

  # Global total per class — used to normalise the scoring so rare and
  # abundant classes contribute on the same scale (sklearn's y_distr).
  class_total = zeros(Int, C)
  for g in unique_groups
    class_total .+= group_class_counts[g]
  end

  # Process order: shuffle, or descending by ‖class_counts‖ for deterministic balancing.
  if alg.shuffle
    Random.shuffle!(rng, unique_groups)
  else
    sort!(unique_groups; by = g -> -sum(abs2, group_class_counts[g]))
  end

  # fold_class_counts[f, c] tracks running totals; pick the fold minimising
  # the variance across folds of class *proportions* (counts / class_total),
  # matching sklearn's `std(y_counts_per_fold / y_distr)` heuristic.
  fold_class_counts = zeros(Int, alg.k, C)
  fold_of = Dict{eltype(unique_groups),Int}()
  for g in unique_groups
    counts = group_class_counts[g]
    best_f = 1
    best_score = Inf
    for f = 1:alg.k
      score = 0.0
      for c = 1:C
        class_total[c] == 0 && continue
        inv_total = 1.0 / class_total[c]
        col_sum = 0.0
        col_sumsq = 0.0
        for ff = 1:alg.k
          v = (fold_class_counts[ff, c] + (ff == f ? counts[c] : 0)) * inv_total
          col_sum += v
          col_sumsq += v * v
        end
        mean_c = col_sum / alg.k
        score += col_sumsq / alg.k - mean_c * mean_c
      end
      if score < best_score
        best_score = score
        best_f = f
      end
    end
    fold_of[g] = best_f
    @views fold_class_counts[best_f, :] .+= counts
  end

  fold_test = [Int[] for _ = 1:alg.k]
  for i = 1:N
    push!(fold_test[fold_of[groups[i]]], i)
  end

  folds_out = Vector{TrainTestSplit{Vector{Int}}}(undef, alg.k)
  for f = 1:alg.k
    test_set = Set(fold_test[f])
    train_idx = [i for i = 1:N if !(i in test_set)]
    folds_out[f] = TrainTestSplit(train_idx, fold_test[f])
  end
  return CrossValidationSplit(folds_out)
end
