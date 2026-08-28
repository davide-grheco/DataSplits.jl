```@meta
CurrentModule = DataSplits
```

# Cross-Validation

Cross-validation (CV) repeatedly fits and evaluates a modelling procedure on different subsets of the available data. In
ordinary `k`-fold cross-validation, the observations are divided into `k` folds: each fold is held out once for
evaluation while the remaining `k-1` folds are used for training.

CV is widely used for both model selection and performance estimation, particularly when the available dataset is too
small to dedicate a large fraction to a fixed validation set [arlotSurvey10](@cite). It reduces dependence on one
arbitrary train/test partition, but it does not remove uncertainty in the performance estimate, and its validity depends
on how the folds are constructed.

When cross-validation is used for model selection or hyperparameter tuning, the same results should not also be treated
as an independent estimate of final performance. A separate evaluation procedure, such as nested cross-validation or an
independent test set, is needed when these two roles must be kept separate.

The appropriate cross-validation strategy is therefore determined by the same principle as any other data split: the
held-out observations should represent the kind of generalisation you want to measure.

## What does cross-validation estimate?

Cross-validation does not directly measure the performance of one particular model fitted to the complete dataset.
Instead, it evaluates models repeatedly fitted to subsets of the available observations. Its estimate therefore
describes the expected performance of the modelling procedure [batesCrossvalidation24](@cite).

The fold structure matters. Ordinary random or shuffled k-fold CV is appropriate when observations can be treated as
exchangeable. When observations are grouped, temporally ordered, spatially dependent, or otherwise correlated, ordinary
CV can place related observations in both training and evaluation folds and produce optimistic performance estimates
[robertsCrossvalidation17](@cite).

Keep the whole modelling procedure inside each fold: Any operation learned from data, including scaling, imputation,
feature selection, dimensionality reduction, and hyperparameter tuning, must be fitted using only the training
observations available in that fold. Applying these steps to the full dataset before cross-validation leaks information
from held-out observations into model development.

All CV strategies return a [`CrossValidationSplit`](@ref) — a collection of folds you can iterate, index, or feed
directly to MLJ.

## Quick reference

| Strategy                                               | Use when                                                                    |
| ------------------------------------------------------ | --------------------------------------------------------------------------- |
| [`KFold`](@ref)                                        | Observations are exchangeable and each should be evaluated once             |
| [`StratifiedKFold`](@ref)                              | Target proportions should remain similar across folds                       |
| [`GroupKFold`](@ref)                                   | Performance should be evaluated on groups absent from training              |
| [`StratifiedGroupKFold`](@ref)                         | Groups must remain intact while approximately preserving target proportions |
| [`ShuffleSplit`](@ref)                                 | You want repeated random train/test partitions with explicit cohort sizes   |
| [`StratifiedShuffleSplit`](@ref)                       | Repeated random partitions should preserve target proportions               |
| [`GroupShuffleSplitCV`](@ref)                          | You want repeated evaluation on unseen groups                               |
| [`RepeatedKFold`](@ref)                                | You want to reduce dependence on one particular random k-fold assignment    |
| [`RepeatedStratifiedKFold`](@ref)                      | The same, while preserving target proportions                               |
| [`BootstrapSplit`](@ref)                               | You want bootstrap training samples with out-of-bag evaluation              |
| [`NestedCV`](@ref)                                     | Model selection and performance estimation must remain separated            |
| [`LeavePOut`](@ref) / [`LeaveOneOut`](@ref)            | Exhaustive holdout of observations is required and computationally feasible |
| [`LeavePGroupsOut`](@ref) / [`LeaveOneGroupOut`](@ref) | Exhaustive evaluation over held-out groups is required                      |
| [`PredefinedSplit`](@ref)                              | Fold membership is determined externally                                    |
| [`TimeSeriesSplit`](@ref)                              | Evaluation must respect temporal ordering                                   |

## K-fold and stratified k-fold

[`KFold`](@ref) divides the observations into `k` approximately equal folds. Each fold is held out once for evaluation
while the remaining `k-1` folds form the training cohort.

When the observations can be treated as exchangeable, their ordering should not carry information relevant to the
prediction problem. If the available data are ordered or the target distribution is uneven, however, naive fold
construction can produce evaluation cohorts with very different compositions.

The Iris dataset provides a simple example. It contains 150 observations from three species, with 50 observations from
each species:

```@example cv
using DataSplits
using RDatasets
using Random

iris = dataset("datasets", "iris")
X = iris[:, 1:4]
y = iris.Species

(setosa = count(==("setosa"), y), versicolor = count(==("versicolor"), y), virginica = count(==("virginica"), y))
```

The observations in the dataset are ordered by species. A deterministic five-fold split therefore produces folds with
very different class compositions:

```@example cv
kfold = partition(X, KFold(5))

classes = unique(y)

function class_counts(cvs, y)
    [
        (
            setosa = count(==("setosa"), y[testindices(fold)]),
            versicolor = count(==("versicolor"), y[testindices(fold)]),
            virginica = count(==("virginica"), y[testindices(fold)]),
        )
        for fold in folds(cvs)
    ]
end

class_counts(kfold, y)
```

One solution is to shuffle observations before constructing the folds:

```@example cv
shuffled = partition(
    X,
    KFold(5; shuffle = true);
    rng = Xoshiro(42),
)

class_counts(shuffled, y)
```

When preserving the target distribution is itself important, [`StratifiedKFold`](@ref) explicitly distributes each class
across the folds:

```@example cv
stratified = partition(
    X,
    StratifiedKFold(5);
    target = y,
)

class_counts(stratified, y)
```

For classification, stratification approximately preserves class proportions across folds. For continuous targets,
DataSplits first groups the response into quantile bins; the number of bins can be controlled with the `bins` argument:

```julia
cvs = partition(X, StratifiedKFold(5; bins = 4); target = y_continuous)
```

Stratification is particularly useful when some classes are rare enough that ordinary fold assignment could leave them
poorly represented in individual evaluation folds. It preserves the target distribution available in the dataset; it
does not however make an imbalanced dataset balanced.

## Group-aware cross-validation

When several observations belong to the same underlying unit, ordinary cross-validation can place observations from the
same group in both training and evaluation folds. If the intended application requires generalisation to previously
unseen groups, this can produce an optimistic estimate of performance [robertsCrossvalidation17](@cite).

The `sleepstudy` dataset contains repeated measurements from individual subjects:

```@example group_cv
using DataSplits
using RDatasets
using Random

sleep = dataset("lme4", "sleepstudy")
X_sleep = sleep[:, [:Days]]
subjects = sleep.Subject

(observations = length(subjects), subjects = length(unique(subjects)))
```

A random k-fold split does not preserve subject membership:

```@example group_cv
ordinary = partition(
    X_sleep,
    KFold(5; shuffle = true);
    rng = Xoshiro(42),
)

function shared_groups(fold, groups)
    intersect(
        Set(groups[trainindices(fold)]),
        Set(groups[testindices(fold)]),
    )
end

[length(shared_groups(fold, subjects)) for fold in folds(ordinary)]
```

[`GroupKFold`] instead assigns each subject to a single fold:

```@example group_cv
grouped = partition(X_sleep, GroupKFold(5); groups = subjects)

[length(shared_groups(fold, subjects)) for fold in folds(grouped)]
```

Every value is zero: no subject appears in both the training and evaluation cohort of the same fold.

Whether this is the appropriate evaluation scheme depends on the deployment problem. Holding out whole groups estimates
generalisation to unseen groups; it is not necessary when future predictions will concern groups already represented
during training.

When both group integrity and approximate target proportions matter, use StratifiedGroupKFold:

```julia
cvs = partition(
    X,
    StratifiedGroupKFold(5);
    target = labels,
    groups = patient_ids,
)
```

## Random Resampling

K-fold CV partitions the observations so that each appears in exactly one evaluation fold. Resampling strategies instead
generate multiple train/test partitions independently, so an observation may be evaluated several times or not at all.

[`ShuffleSplit`](@ref) is useful when you want repeated random partitions and explicit control over the size of each
cohort:

```@example cv
shuffled = partition(X, ShuffleSplit(10);
    train = 0.8,
    test = 0.2,
    rng = Xoshiro(42),
)

[
    (
        train = length(trainindices(fold)),
        test = length(testindices(fold)),
    )
    for fold in folds(shuffled)
]
```

**[`StratifiedShuffleSplit`](@ref)** additionally preserves the target distribution within each resample:

```@example cv
stratified_shuffled = partition(X, StratifiedShuffleSplit(10);
    target = y,
    train = 0.8,
    test = 0.2,
    rng = Xoshiro(42),
)

length(folds(stratified_shuffled))
```

For grouped observations, [`GroupShuffleSplitCV`](@ref) repeatedly samples whole groups rather than individual
observations:

```@example group_cv
group_shuffled = partition(X_sleep, GroupShuffleSplitCV(10);
    groups = subjects,
    train = 0.8,
    test = 0.2,
    rng = Xoshiro(42),
)

[length(shared_groups(fold, subjects)) for fold in folds(group_shuffled)]
```

Because groups are indivisible, requested cohort proportions may not be achievable exactly.

## Repeated KFold

[`RepeatedKFold`](@ref) runs KFold `n_repeats` times with different shuffled assignments, producing `k × n_repeats`
folds. This reduces the variance of the performance estimate compared to a single k-fold run.

```@example cv
repeated = partition(
    X,
    RepeatedKFold(5; n_repeats = 10);
    rng = Xoshiro(42),
)

length(folds(repeated))
```

[`RepeatedStratifiedKFold`](@ref) does the same with stratification:

```julia
cvs = partition(X, RepeatedStratifiedKFold(5; n_repeats = 10);
                target = labels, rng = MersenneTwister(42))
```

## Fold variability is not a standard error

Scores obtained from different folds are correlated because their training sets overlap. The variation between fold
scores should therefore not automatically be interpreted as the sampling uncertainty of the overall cross-validation
estimate. Repeating cross-validation provides information about sensitivity to the partition, but it does not make the
individual fold results independent.

## Bootstrap

[`BootstrapSplit`](@ref) constructs each training cohort by drawing `N` observations with replacement. Observations not
drawn in a bootstrap sample form its out-of-bag (OOB) evaluation cohort.

A bootstrap training cohort always contains `N` draws, but some observations occur more than once. The OOB observations
provide an evaluation cohort for each bootstrap sample. Use [`ShuffleSplit`](@ref) instead when every training cohort
should contain unique observations.

```@example cv
bootstrap = partition(
    X,
    BootstrapSplit(5);
    rng = Xoshiro(42),
)

[
    (
        draws = length(trainindices(fold)),
        unique_train = length(unique(trainindices(fold))),
        out_of_bag = length(testindices(fold)),
    )
    for fold in folds(bootstrap)
]

```

## Leave-p-out and leave-group-out

[`LeaveOneOut`](@ref) holds out one observation at a time, producing `N` folds. [`LeavePOut`](@ref) generalises this to
every possible combination of p held-out observations:

The number of folds produced by leave-p-out is combinatorial, so exhaustive evaluation becomes computationally expensive
very quickly.

```julia
cvs = partition(X, LeaveOneOut())  # N folds
cvs = partition(X, LeavePOut(3))   # binomial(N, 3) folds — use only for small N
```

More folds do not automatically imply a better performance estimate. Leave-one-out trains on almost the complete dataset
in every fold, but it can be computationally expensive and may have undesirable statistical properties for model
selection. The appropriate number of folds depends on the dataset and the modelling problem.

[`LeaveOneGroupOut`](@ref) / [`LeavePGroupsOut`](@ref) apply the same idea to groups:

```julia
cvs = partition(X, LeaveOneGroupOut(); groups = batch_ids)  # one batch held out per fold
cvs = partition(X, LeavePGroupsOut(2); groups = site_ids)   # binomial(n_groups, 2) folds
```

These strategies are useful when performance must be examined systematically across held-out groups, such as individual
sites, batches, subjects, or experimental units.

## Nested cross-validation

[`NestedCV`](@ref) combines an outer CV (for unbiased performance estimation) with an inner CV (for hyperparameter
tuning). For each outer fold the inner CV is applied to the outer training cohort; inner indices are remapped to the
global `1:N` space.

Cross-validation is often used to choose hyperparameters, features, preprocessing steps, or even the model family
itself. Once the CV results have been used to make those choices, reporting the same results as an independent estimate
of final performance can be optimistic.

[`NestedCV`](@ref) separates model selection from performance estimation. For each outer fold:

- The outer evaluation cohort is held out;
- The inner folds are constructed using only the outer training cohort;
- Modelling choices are made using the inner folds;
- The selected procedure is refitted using the complete outer training cohort;
- Performance is evaluated on the outer evaluation cohort.

The outer folds therefore estimate the performance of the complete model-selection procedure and are an alternative to a
fixed test set evaluation.

```julia
cvs = partition(X, NestedCV(KFold(5), KFold(3)))

for outer_fold in folds(cvs)
    inner = innerfolds(outer_fold)

    for inner_fold in folds(inner)
        X_tr, y_tr = trainview(inner_fold, X, y)
        X_val, y_val = testview(inner_fold, X, y)

        # Fit candidate procedure on X_tr, y_tr
        # Evaluate candidate on X_val, y_val
    end

    X_tr_outer, y_tr_outer = trainview(outer_fold, X, y)
    X_te_outer, y_te_outer = testview(outer_fold, X, y)

    # Refit the selected procedure on X_tr_outer, y_tr_outer
    # Evaluate on X_te_outer, y_te_outer
endnd
```

The inner strategy must be a non-resampling [`AbstractCVStrategy`](@ref). Resampling strategies such as
[`ShuffleSplit`](@ref), [`StratifiedShuffleSplit`](@ref), and [`GroupShuffleSplitCV`](@ref) require caller-specified
cohort sizes that [`NestedCV`](@ref) does not propagate.

Stratified and group-aware strategies can be used for both the inner and outer folds when the corresponding structure
must be preserved during both model selection and performance estimation:

```julia
cvs = partition(X, NestedCV(StratifiedKFold(5), StratifiedKFold(3));
                target = labels)
```

## Predefined fold assignments

[`PredefinedSplit`](@ref) is useful when the evaluation folds are determined externally, for example by an experimental
design, benchmark definition, or previously published partition.

The fold assignment vector specifies which observations belong to each evaluation fold. Negative assignments indicate
observations that always remain in training and are never evaluated:

```julia
# 3 folds: obs 1-20 test in fold 0, obs 21-40 in fold 1, obs 41-60 in fold 2.
test_fold = [fill(0, 20); fill(1, 20); fill(2, 20)]
cvs = partition(X, PredefinedSplit(test_fold))

# Hold-out: last 10 observations are always in train, never tested.
test_fold = [fill(0, 40); fill(-1, 10)]
cvs = partition(X, PredefinedSplit(test_fold))
```

## Time-dependent data

Ordinary k-fold cross-validation is generally inappropriate when the prediction problem has a temporal direction or when
nearby observations are strongly dependent. In these settings, fold construction should reflect the information
available at prediction time or otherwise account for the temporal dependence structure.

[`TimeSeriesSplit`](@ref), [`BlockedCV`](@ref), [`PurgedKFold`](@ref), and related strategies are described in
[Time-Series Splitting](06-time-series.md).
