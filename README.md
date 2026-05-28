# DataSplits.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://davide-grheco.github.io/DataSplits.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://davide-grheco.github.io/DataSplits.jl/dev)
[![Test workflow status](https://github.com/davide-grheco/DataSplits.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/davide-grheco/DataSplits.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/davide-grheco/DataSplits.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/davide-grheco/DataSplits.jl)
[![Docs workflow Status](https://github.com/davide-grheco/DataSplits.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/davide-grheco/DataSplits.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)
[![DOI](https://zenodo.org/badge/1020676159.svg)](https://doi.org/10.5281/zenodo.20426864)

A Julia library of train/test and cross-validation splitting strategies for cases
where random splitting misleads — small datasets, regression over continuous targets,
grouped observations, time series, molecular or geospatial data.

One entry point covers everything: `partition`.

## Installation

```julia
using Pkg
Pkg.add("DataSplits")
```

## Quick start

```julia
using DataSplits

# Diversity-based split — training set that uniformly covers the feature space.
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
| Cover feature space (maximin) | `KennardStoneSplit` / `LazyKennardStoneSplit` |
| Cover features + target jointly | `SPXYSplit`, `MDKSSplit` |
| Diversity selection (subsample) | `OptiSimSplit`, `MinimumDissimilaritySplit`, `MaximumDissimilaritySplit` |
| Kennard–Stone + random swap | `MoraisLimaMartinSplit` |
| Group-aware train/test | `GroupShuffleSplit`, `GroupStratifiedSplit` |
| Time-ordered train/test | `TimeSplit` (`TimeSplitOldest`, `TimeSplitNewest`) |
| Train on extreme target values | `TargetPropertySplit` (`TargetPropertyHigh`, `TargetPropertyLow`) |
| Random baseline | `RandomSplit` |
| Plain k-fold | `KFold` |
| Stratified k-fold | `StratifiedKFold` |
| Group k-fold | `GroupKFold`, `StratifiedGroupKFold` |
| Leave-group-out | `LeaveOneGroupOut`, `LeavePGroupsOut` |
| Time-series CV | `TimeSeriesSplit`, `BlockedCV`, `PurgedKFold` |
| Resampling CV | `ShuffleSplit`, `StratifiedShuffleSplit`, `GroupShuffleSplitCV`, `BootstrapSplit` |
| Repeated CV | `RepeatedKFold`, `RepeatedStratifiedKFold` |
| Nested CV | `NestedCV` |
| Predefined fold assignments | `PredefinedSplit` |
| Leave-p-out | `LeavePOut`, `LeaveOneOut` |
| Cluster assignment | `sphere_exclusion` |

## Conventions

- Matrices follow the Julia ML convention: **columns are samples, rows are features**.
  Tables.jl inputs (e.g. `DataFrame`) use rows as samples and are converted internally.
- Custom containers must implement `MLUtils.numobs` and `MLUtils.getobs`.
- Cohort sizes (`train`, `validation`, `test`) are set on `partition`, not on the strategy.
  They accept integer counts, integer percentages (summing to 100), or `(0, 1)` fractions
  (summing to 1.0).
- All strategies accept an `rng` keyword for reproducibility.
