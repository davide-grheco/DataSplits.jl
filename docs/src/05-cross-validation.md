```@meta
CurrentModule = DataSplits
```

# Cross-Validation

Cross-validation (CV) estimates model performance more reliably than a single
train/test split by rotating the test window across the data. DataSplits provides
the full standard catalogue of CV strategies, all accessible through
[`partition`](@ref) with a uniform interface.

All CV strategies return a [`CrossValidationSplit`](@ref) — a collection of folds
you can iterate, index, or feed directly to MLJ.

## Quick reference

| Strategy | Key property |
| --- | --- |
| [`KFold`](@ref) | Plain k-fold; deterministic or shuffled |
| [`StratifiedKFold`](@ref) | Preserves class / quantile-bin proportions per fold |
| [`GroupKFold`](@ref) | No group spans two folds |
| [`StratifiedGroupKFold`](@ref) | Group integrity + class balance |
| [`ShuffleSplit`](@ref) | Independent random resamples; caller sets cohort sizes |
| [`StratifiedShuffleSplit`](@ref) | Stratified resampling |
| [`GroupShuffleSplitCV`](@ref) | Group-aware resampling |
| [`RepeatedKFold`](@ref) | KFold run multiple times with different shuffles |
| [`RepeatedStratifiedKFold`](@ref) | Same, stratified |
| [`BootstrapSplit`](@ref) | Bootstrap resampling; OOB as test |
| [`NestedCV`](@ref) | Outer CV for evaluation, inner CV for hyperparameter tuning |
| [`LeavePOut`](@ref) / [`LeaveOneOut`](@ref) | Every combination of p observations as test |
| [`LeavePGroupsOut`](@ref) / [`LeaveOneGroupOut`](@ref) | Every combination of p groups as test |
| [`PredefinedSplit`](@ref) | Caller provides fold assignments |
| [`TimeSeriesSplit`](@ref) | Time-aware; see also the [Time Series](06-time-series.md) page |

## The iteration pattern

```julia
cvs = partition(X, KFold(5))

for (X_tr, X_te) in splitview(cvs, X)
    fit!(model, X_tr)
    score = evaluate(model, X_te)
end

# MLJ integration.
using MLJ
mach = machine(model, X, y)
evaluate!(mach; resampling = rowpairs(cvs), measure = accuracy)
```

## Plain KFold

[`KFold`](@ref) divides the data into `k` roughly equal folds. Each fold takes a
turn as the test set; the remaining `k-1` folds form the training set.

```julia
# Deterministic split (default).
cvs = partition(X, KFold(5))

# Shuffle observations before folding for a different assignment each time.
cvs = partition(X, KFold(5; shuffle = true); rng = MersenneTwister(42))
```

Fold sizes differ by at most one observation: the first `N mod k` folds are one
sample larger.

## Stratified KFold

[`StratifiedKFold`](@ref) distributes each class (or quantile bin for continuous
targets) round-robin across the `k` folds so every fold has nearly the same class
proportions as the full dataset.

```julia
# Classification: class labels as target.
cvs = partition(X, StratifiedKFold(5); target = labels)

# Regression: continuous target binned into 10 quantile groups by default.
cvs = partition(X, StratifiedKFold(5); target = y)

# Fewer bins for sparse or discrete-heavy targets.
cvs = partition(X, StratifiedKFold(5; bins = 4); target = y)
```

Use [`StratifiedKFold`](@ref) instead of [`KFold`](@ref) whenever the class
distribution is imbalanced or the dataset is small.

## Group-aware KFold

[`GroupKFold`](@ref) assigns entire groups to single folds — no group ever appears
in both the train and test cohort of the same fold. This is the standard choice for
datasets with natural grouping (patients, molecular scaffolds, experimental batches).

```julia
cvs = partition(X, GroupKFold(5); groups = patient_ids)

# Shuffle group assignment order (different fold compositions each run).
cvs = partition(X, GroupKFold(5; shuffle = true);
                groups = patient_ids, rng = MersenneTwister(42))
```

For the most demanding case — group integrity *and* class balance —
use [`StratifiedGroupKFold`](@ref):

```julia
cvs = partition(X, StratifiedGroupKFold(5);
                target = labels, groups = patient_ids)
```

## Leave-p-out and leave-group-out

[`LeaveOneOut`](@ref) produces `N` folds, each with a single test observation.
Exhaustive but expensive for large datasets.

```julia
cvs = partition(X, LeaveOneOut())  # N folds
cvs = partition(X, LeavePOut(3))   # binomial(N, 3) folds — use only for small N
```

[`LeaveOneGroupOut`](@ref) / [`LeavePGroupsOut`](@ref) are the group-aware
analogues — every combination of one (or `p`) groups takes a turn as the test cohort.

```julia
cvs = partition(X, LeaveOneGroupOut(); groups = batch_ids)  # one batch held out per fold
cvs = partition(X, LeavePGroupsOut(2); groups = site_ids)   # binomial(n_groups, 2) folds
```

## Resampling strategies

**[`ShuffleSplit`](@ref)** produces `n_splits` independent random resamples, each
sized by the caller. Unlike KFold, a single observation can appear in test in
multiple folds.

```julia
cvs = partition(X, ShuffleSplit(10); train = 0.8, test = 0.2)
cvs = partition(X, ShuffleSplit(10); train = 0.8, test = 0.2,
                rng = MersenneTwister(42))
```

**[`StratifiedShuffleSplit`](@ref)** adds class balancing per resample:

```julia
cvs = partition(X, StratifiedShuffleSplit(10); target = labels,
                train = 0.8, test = 0.2)
```

**[`GroupShuffleSplitCV`](@ref)** is the group-aware resampling variant — groups
are added whole, so the actual train size may overshoot slightly:

```julia
cvs = partition(X, GroupShuffleSplitCV(10);
                groups = patient_ids, train = 0.8, test = 0.2)
```

## Bootstrap

[`BootstrapSplit`](@ref) draws `N` observations *with replacement* as the training
set; the observations never drawn form the out-of-bag (OOB) test set. On average
about 63.2% of unique observations land in train; the rest form the OOB test.

```julia
cvs = partition(X, BootstrapSplit(50); rng = MersenneTwister(42))

for (X_tr, X_te) in splitview(cvs, X)
    # X_tr has N observations, with duplicates — this is by design
    # X_te is the OOB set (unique observations not drawn in this bootstrap)
end
```

Use [`ShuffleSplit`](@ref) if you need unique indices only.

## Repeated KFold

[`RepeatedKFold`](@ref) runs KFold `n_repeats` times with a fresh random shuffle
each time, producing `k × n_repeats` folds. This reduces the variance of the
performance estimate compared to a single k-fold run.

```julia
cvs = partition(X, RepeatedKFold(5; n_repeats = 10);
                rng = MersenneTwister(42))  # 50 folds total
```

[`RepeatedStratifiedKFold`](@ref) does the same with stratification:

```julia
cvs = partition(X, RepeatedStratifiedKFold(5; n_repeats = 10);
                target = labels, rng = MersenneTwister(42))
```

## Nested cross-validation

[`NestedCV`](@ref) combines an outer CV (for unbiased performance estimation) with
an inner CV (for hyperparameter tuning). For each outer fold the inner CV is applied
to the outer training cohort; inner indices are remapped to the global `1:N` space.

```julia
cvs = partition(X, NestedCV(KFold(5), KFold(3)))

for outerfold in folds(cvs)
    X_tr_outer, X_te_outer = splitdata(outerfold, X)

    for (X_tr, X_val) in splitview(innerfolds(outerfold), X)
        # Tune hyperparameters on (X_tr, X_val)
    end
    # Refit best model on full X_tr_outer, score on X_te_outer
end
```

!!! note "Inner strategy restriction"
    The inner strategy must be a non-resampling `AbstractCVStrategy`. Passing a
    resampling strategy (`ShuffleSplit`, `StratifiedShuffleSplit`,
    `GroupShuffleSplitCV`) as the inner argument raises a `SplitParameterError`
    at construction time, because those strategies require caller-specified cohort
    sizes that `NestedCV` does not propagate.

Stratified and group-aware strategies work as both outer and inner:

```julia
cvs = partition(X, NestedCV(StratifiedKFold(5), StratifiedKFold(3));
                target = labels)
```

## Predefined fold assignments

[`PredefinedSplit`](@ref) lets you supply the fold assignment vector directly.
Observations with a negative assignment are always placed in train.

```julia
# 3 folds: obs 1-20 test in fold 0, obs 21-40 in fold 1, obs 41-60 in fold 2.
test_fold = [fill(0, 20); fill(1, 20); fill(2, 20)]
cvs = partition(X, PredefinedSplit(test_fold))

# Hold-out: last 10 observations are always in train, never tested.
test_fold = [fill(0, 40); fill(-1, 10)]
cvs = partition(X, PredefinedSplit(test_fold))
```
