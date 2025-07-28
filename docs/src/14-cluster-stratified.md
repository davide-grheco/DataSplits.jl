```@meta
CurrentModule = DataSplits
```

# Cluster Stratified Split

## Overview

Cluster Stratified Split is a train/test splitting strategy that ensures each cluster (as determined by a clustering algorithm) is split into train and test sets according to a specified allocation method and user-defined train fraction.

## Allocation Methods

- **Equal allocation**: Randomly selects a fixed number `n` of samples from each cluster, then splits those into train/test according to the user-specified `frac` (fraction for train set). The rest of the cluster is unused.
- **Proportional allocation**: Uses all samples in each cluster, splits them into train/test according to the user-specified `frac` (fraction for train set).
- **Neyman allocation**: Randomly selects a quota from each cluster (proportional to cluster size and mean standard deviation of features), then splits those into train/test according to the user-specified `frac` (fraction for train set). The rest of the cluster is unused.

## When to Use

- When you want to preserve cluster structure in both train and test sets
- For grouped or clustered data where stratification is important

## When Not to Use

- When clusters are very imbalanced in size (may affect stratification)
- For unstructured data with no meaningful groups

## Arguments

- `clusters`: ClusteringResult (from Clustering.jl)
- `allocation`: `:equal`, `:proportional`, or `:neyman`
- `n`: Number of samples per cluster (for `:equal` and `:neyman`)
- `frac`: Fraction of selected samples to use for train (rest go to test)

## Usage

```julia
using DataSplits, Clustering
clusters = sphere_exclusion(X; radius=0.3)
splitter = ClusterStratifiedSplit(clusters, :proportional; frac=0.7)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```

## Notes/Limitations

- If `n` is greater than the cluster size, all samples in the cluster are used
- For `:proportional`, all samples are always used
- For `frac=1.0`, all selected samples go to train; for `frac=0.0`, all go to test
- The split is performed per cluster; rounding is handled so that the train set always gets the larger share when the split is not exact

## API Reference

- [`ClusterStratifiedSplit`](@ref)
- [`split`](@ref)
- [`splitdata`](@ref)
