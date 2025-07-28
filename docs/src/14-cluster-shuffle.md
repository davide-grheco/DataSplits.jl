```@meta
CurrentModule = DataSplits
```

# Cluster Shuffle Split

## Overview

Cluster Shuffle Split is a group-aware splitting strategy. It either takes a precomputed clustering result or a function to generate one, then shuffles the cluster labels and accumulates whole clusters into the training set until the desired fraction is reached. This approach is ideal for grouped or clustered data where splitting within groups would break structure or introduce leakage.

## When to Use

- When your data has natural groups or clusters (e.g., patients, molecules, batches)
- When you want to avoid splitting groups across train/test

## When Not to Use

- When clusters are very imbalanced in size (fraction control is coarse)
- For unstructured data with no meaningful groups

## Arguments

- `res` or `f,data`: Clustering result or function
- `frac`: Fraction of samples to use for training (0 < frac < 1)
- `rng`: Optional RNG

## Usage

```julia
using DataSplits, Clustering
clusters = sphere_exclusion(X; radius=0.3)
splitter = ClusterShuffleSplit(clusters, 0.8)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```

## Notes/Limitations

- Fraction control is coarse; may overshoot/undershoot target
- Cluster sizes may vary widely

## API Reference

- [`ClusterShuffleSplit`](@ref)
- [`split`](@ref)
- [`splitdata`](@ref)
