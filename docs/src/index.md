```@meta
CurrentModule = DataSplits
```

# DataSplits

DataSplits is a Julia library of **rational train/test/CV splitting strategies**, intended for cases where random selection overestimates model performance — small datasets, structured data, regression with non-uniform targets, grouped observations, time series.

The package exposes a single entry point, [`partition`](@ref), and a catalogue of strategies covering distance-based, group-aware, target-property, temporal, random / resampling, and cross-validation splits.

## Quick start

```julia
using DataSplits

# Single train/test split (Kennard–Stone).
res = partition(X, KennardStoneSplit(); train = 0.8, test = 0.2)
X_train, X_test = splitdata(res, X)

# Joint X / y diversity (SPXY).
res = partition(X, SPXYSplit(); target = y, train = 80, test = 20)
X_train, X_test = splitdata(res, X)

# Train / validation / test in one call.
res = partition(X, RandomSplit(), KennardStoneSplit();
                train = 70, validation = 10, test = 20)
X_tr, X_val, X_te = splitdata(res, X)

# Group-aware cross-validation.
cvs = partition(X, GroupKFold(5); groups = patient_ids)
for (X_tr, X_te) in splitview(cvs, X)
    # train and evaluate
end
```

See [Getting Started](02-getting-started.md) for the full API.

## Cheat sheet

| Task                                | Strategy                                         |
|-------------------------------------|--------------------------------------------------|
| Maximin on *X*                      | [`KennardStoneSplit`](@ref) (alias `CADEXSplit`) |
| Maximin on *X* + *y*                | [`SPXYSplit`](@ref), [`MDKSSplit`](@ref)         |
| Diversity selection                 | [`OptiSimSplit`](@ref), [`MinimumDissimilaritySplit`](@ref), [`MaximumDissimilaritySplit`](@ref) |
| Kennard–Stone with swap             | [`MoraisLimaMartinSplit`](@ref)                  |
| Group-aware split                   | [`GroupShuffleSplit`](@ref), [`GroupStratifiedSplit`](@ref) |
| Time-based split                    | [`TimeSplit`](@ref) (`TimeSplitOldest`, `TimeSplitNewest`) |
| Target-property split               | [`TargetPropertySplit`](@ref) (`TargetPropertyHigh`, `TargetPropertyLow`) |
| Random baseline                     | [`RandomSplit`](@ref)                            |
| Plain k-fold                        | [`KFold`](@ref)                                  |
| Group k-fold                        | [`GroupKFold`](@ref), [`LeaveOneGroupOut`](@ref), [`LeavePGroupsOut`](@ref) |
| Stratified k-fold                   | [`StratifiedKFold`](@ref), [`StratifiedGroupKFold`](@ref) |
| Time-series CV                      | [`TimeSeriesSplit`](@ref), [`BlockedCV`](@ref), [`PurgedKFold`](@ref) |
| Resampling CV                       | [`ShuffleSplit`](@ref), [`StratifiedShuffleSplit`](@ref), [`GroupShuffleSplitCV`](@ref), [`BootstrapSplit`](@ref) |
| Repeated CV                         | [`RepeatedKFold`](@ref), [`RepeatedStratifiedKFold`](@ref) |
| Nested CV                           | [`NestedCV`](@ref)                               |
| Predefined fold assignments         | [`PredefinedSplit`](@ref)                        |
| Leave-p-out                         | [`LeavePOut`](@ref), [`LeaveOneOut`](@ref)       |
| Clustering helper                   | [`sphere_exclusion`](@ref)                       |

## Conventions

- Matrices follow the Julia ML convention: **columns are samples, rows are features**. Tables.jl inputs (e.g. `DataFrame`) use rows as samples and are converted internally.
- Custom containers must implement `MLUtils.numobs` and `MLUtils.getobs`.
- Cohort sizes (`train`, `validation`, `test`) live on `partition`, not on the strategy. They accept integer counts, integer percentages summing to 100, or `(0, 1)` fractions summing to 1.

## Navigation

- [Getting Started](02-getting-started.md) — `partition` in three forms, slot resolution, materialising splits.
- [Core API Reference](03-core-api-reference.md) — full signatures, result types, accessors, traits, exceptions.
- [Algorithms](04-algorithms-overview.md) — the catalogue of built-in strategies.
- [Extending DataSplits](05-extending-data-splits.md) — adding a custom strategy.
- [Reference](95-reference.md) — auto-generated docstring index.
