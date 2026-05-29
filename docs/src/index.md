```@meta
CurrentModule = DataSplits
```

# DataSplits.jl

DataSplits is a Julia library of train/test and cross-validation splitting strategies
for cases where random selection misleads — small datasets, regression over continuous
targets, grouped observations, time series, molecular or geospatial data.

One entry point covers everything: [`partition`](@ref).

## Installation

```julia
using Pkg
Pkg.add("DataSplits")
```

## Quick start

```julia
using DataSplits

# Diversity-based split — training set that covers the full feature space.
res = partition(X, KennardStoneSplit(); train = 0.8, test = 0.2)
X_train, X_test = splitdata(res, X)

# Cover features and target jointly (SPXY).
res = partition(X, SPXYSplit(); target = y, train = 80, test = 20)

# Train / validation / test in one call.
res = partition(X, RandomSplit(), KennardStoneSplit();
                train = 70, validation = 10, test = 20)
X_tr, X_val, X_te = splitdata(res, X)

# Group-aware k-fold: no patient, scaffold, or batch spans two folds.
cvs = partition(X, GroupKFold(5); groups = patient_ids)
for (X_tr, X_te) in splitview(cvs, X)
    fit!(model, X_tr)
    evaluate(model, X_te)
end
```

## Strategy catalogue

| Task | Strategy |
| --- | --- |
| Cover feature space (maximin) | [`KennardStoneSplit`](@ref) / [`LazyKennardStoneSplit`](@ref) |
| Simultaneous train+test coverage (Duplex) | [`DuplexSplit`](@ref), [`LazyDuplexSplit`](@ref) |
| Cover features + target jointly | [`SPXYSplit`](@ref), [`MDKSSplit`](@ref) |
| Field-strength split | [`FieldStrengthSplit`](@ref) |
| Spectral cluster split | [`SpectralSplit`](@ref) |
| Diversity selection (subsample) | [`OptiSimSplit`](@ref), [`MinimumDissimilaritySplit`](@ref), [`MaximumDissimilaritySplit`](@ref) |
| Kennard–Stone + random swap | [`MoraisLimaMartinSplit`](@ref) |
| Group-aware train/test | [`GroupShuffleSplit`](@ref), [`GroupStratifiedSplit`](@ref) |
| Time-ordered train/test | [`TimeSplit`](@ref) (`TimeSplitOldest`, `TimeSplitNewest`) |
| Train on extreme target values | [`TargetPropertySplit`](@ref) (`TargetPropertyHigh`, `TargetPropertyLow`) |
| Random baseline | [`RandomSplit`](@ref) |
| Plain k-fold | [`KFold`](@ref) |
| Stratified k-fold | [`StratifiedKFold`](@ref) |
| Group k-fold | [`GroupKFold`](@ref), [`StratifiedGroupKFold`](@ref) |
| Leave-group-out | [`LeaveOneGroupOut`](@ref), [`LeavePGroupsOut`](@ref) |
| Time-series CV | [`TimeSeriesSplit`](@ref), [`BlockedCV`](@ref), [`PurgedKFold`](@ref), [`CombinatorialPurgedKFold`](@ref) |
| Resampling CV | [`ShuffleSplit`](@ref), [`StratifiedShuffleSplit`](@ref), [`GroupShuffleSplitCV`](@ref), [`BootstrapSplit`](@ref) |
| Repeated CV | [`RepeatedKFold`](@ref), [`RepeatedStratifiedKFold`](@ref) |
| Nested CV | [`NestedCV`](@ref) |
| Predefined fold assignments | [`PredefinedSplit`](@ref) |
| Leave-p-out | [`LeavePOut`](@ref), [`LeaveOneOut`](@ref) |
| Cluster assignment | [`sphere_exclusion`](@ref) |

## Conventions

- Matrices follow the Julia ML convention: **columns are samples, rows are features**.
  Tables.jl inputs (e.g. `DataFrame`) use rows as samples and are converted internally.
- Custom containers must implement `MLUtils.numobs` and `MLUtils.getobs`.
- Cohort sizes (`train`, `validation`, `test`) are set on `partition`, not on the strategy.
  They accept integer counts, integer percentages summing to 100, or `(0,1)` fractions
  summing to 1.
