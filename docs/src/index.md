```@meta
CurrentModule = DataSplits
```

# DataSplits.jl

DataSplits.jl provides a common interface for splitting data for model training, selection, and performance evaluation.

It includes random and resampling methods, cross-validation, group-aware and time-aware strategies, distance-based
sample selection, and extrapolative splits.

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

If you are new to data splitting, start with [Why Splitting Matters](01-introduction.md).

For a hands-on introduction to the API, continue to [Getting Started](02-getting-started.md).
