```@meta
CurrentModule = DataSplits
```

# Sphere Exclusion Split

## Overview

Sphere Exclusion Split is a clustering-based splitting strategy. It works by iteratively picking an unassigned sample and forming a cluster of all points within a specified radius (in normalized distance). This process repeats until all samples are assigned to clusters. Clusters are then assigned to the training set until the desired fraction is reached. This method is especially useful for spatial or similarity-based data, where you want to avoid splitting local neighborhoods.

## When to Use

- For spatial, chemical, or biological data where locality matters
- When you want to avoid splitting similar samples across train/test

## When Not to Use

- When you do not know how to set a meaningful radius
- For data with no meaningful distance metric

## Arguments

- `frac`: Fraction of samples to use for training (0 < frac < 1)
- `radius`: Normalized distance threshold [0,1]
- `metric`: Distance metric (default: Euclidean)

## Usage

```julia
using DataSplits
splitter = SphereExclusionSplit(0.7; radius=0.2)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```

## Notes/Limitations

- Sensitive to radius; cluster sizes may be uneven
- May not work well for high-dimensional or non-metric data

## API Reference

- [`sphere_exclusion`](@ref)
- [`split`](@ref)
- [`splitdata`](@ref)

## References

- [Sphere Exclusion Clustering](https://doi.org/10.1021/ci00057a005)
