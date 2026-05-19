```@meta
CurrentModule = DataSplits
```

# Why Splitting Matters

You've built a model and you need to know how well it generalises to new data. The
answer depends entirely on how you split your data: a bad split gives you a falsely
optimistic error estimate, and you only discover this after deployment.

This page explains the common failure modes of naive splits and shows which strategy
to reach for in each situation.

## The problem with random splits

A random split assigns each observation independently to train or test with equal
probability. For large, independently and identically distributed datasets this works
well. For the datasets that practitioners actually work with — small, structured,
grouped, or temporal — it routinely fails.

### Small datasets and interpolation bias

Suppose you have 100 chemical compounds and you randomly assign 80 to training
and 20 to test. Because the random split distributes points across the feature space
without regard to coverage, the 20 test points tend to land *inside* the training
cloud. The model is effectively asked to interpolate between its own training points,
not to extrapolate to unseen chemistry. The measured error looks great; real
predictions on novel compounds look much worse.

A **diversity-based split** (Kennard–Stone, SPXY, OptiSim) instead selects training
points that spread uniformly across the feature space, pushing test points toward
the boundaries of the distribution and giving a more honest estimate of
generalisation.

### Group leakage

Many datasets have natural groups: multiple samples from the same patient, assays
from the same batch, molecules sharing the same scaffold, observations from the same
geographic site. Within a group, measurements are correlated — they share hidden
confounders.

If a random split places some samples from *patient 17* in training and others in
test, the model can implicitly learn patient-specific patterns and score well on test
without having learned anything generally transferable. This is **group leakage**.

A **group-aware split** ([`GroupShuffleSplit`](@ref), [`GroupKFold`](@ref)) ensures
that every sample from a given group ends up in the same cohort.

### Temporal leakage

Time series data has a direction: the future cannot cause the past. A random split
ignores this — if the test set contains data from January and the training set
contains data from February, the model has access to information it could never have
in production.

A **time-aware split** ([`TimeSeriesSplit`](@ref), [`BlockedCV`](@ref),
[`TimeSplit`](@ref)) always trains on the past and evaluates on the future.

### Extrapolation vs. interpolation

Sometimes the scientific question itself requires extrapolation. If you are building
a model to predict molecules outside the training domain, you should test *on*
out-of-domain molecules, not on molecules drawn from the same distribution as
training.

[`TargetPropertySplit`](@ref) and target-aware strategies like [`SPXYSplit`](@ref)
let you deliberately construct such challenge sets.

## Choosing a strategy

| Situation | Recommended strategy |
| --- | --- |
| Small dataset, continuous features | [`KennardStoneSplit`](@ref) or [`SPXYSplit`](@ref) |
| Large dataset, memory constrained | [`LazyKennardStoneSplit`](@ref), [`LazyOptiSimSplit`](@ref) |
| Observations belong to groups (patients, scaffolds, batches) | [`GroupShuffleSplit`](@ref) / [`GroupKFold`](@ref) |
| Time-ordered data | [`TimeSeriesSplit`](@ref) for CV, [`TimeSplit`](@ref) for a single split |
| Financial / autocorrelated time series | [`BlockedCV`](@ref) or [`PurgedKFold`](@ref) |
| Balanced class distribution across folds | [`StratifiedKFold`](@ref) |
| Hyperparameter tuning + unbiased evaluation | [`NestedCV`](@ref) |
| Large i.i.d. dataset, need many resamples | [`ShuffleSplit`](@ref) or [`BootstrapSplit`](@ref) |
| Need a reproducible random baseline | [`RandomSplit`](@ref) |

## What DataSplits provides

- **Distance-based strategies** — Kennard–Stone, SPXY, MDKS, OptiSim, Minimum and
  Maximum Dissimilarity — for small-to-medium datasets where training-set coverage of
  the feature space matters.
- **Cross-validation strategies** — KFold and all its stratified, group-aware, and
  time-series variants — for model evaluation and hyperparameter tuning.
- **Group-aware strategies** — split or fold data while keeping entire groups intact.
- **Time-series strategies** — chronological train/test and cross-validation that
  respect temporal order.
- **A uniform API** — every strategy is called through [`partition`](@ref), cohort
  sizes are always specified at call time, and results are always indices you can
  apply to any container.

Continue to [Getting Started](02-getting-started.md) for the first hands-on example.
