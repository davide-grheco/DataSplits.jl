```@meta
CurrentModule = DataSplits
```

# Kennard–Stone Split

## Overview

The Kennard–Stone algorithm (also known as **CADEX** in some literature) is a deterministic method for selecting a representative training set from a dataset. It iteratively chooses the sample that is farthest (in feature space) from all previously selected samples, starting from the most distant pair. This ensures that the training set covers the full range of the feature space, making it especially useful for rational dataset splitting.

*CADEX* stands for **Computer Aided Design of Experiments** and is an alias for the Kennard–Stone algorithm in DataSplits.

## How it works

1. Compute the pairwise distance matrix for all samples.
2. Select the two samples that are farthest apart as the initial training set.
3. Iteratively add the sample that is farthest from the current training set (i.e., has the largest minimum distance to any selected sample).
4. Continue until the desired number of training samples is reached.

## Arguments

- `frac`: Fraction of samples to use for training (0 < frac < 1)
- `metric`: Distance metric (default: Euclidean)

## Usage

```julia
using DataSplits, Distances
splitter = KennardStoneSplit(0.8)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```

## API Reference

- [`KennardStoneSplit`](@ref)
- [`LazyKennardStoneSplit`](@ref)
- [`split`](@ref)
- [`splitdata`](@ref)

## References

Kennard, R. W.; Stone, L. A. Computer Aided Design of Experiments. Technometrics 1969, 11 (1), 137–148. <https://doi.org/10.1080/00401706.1969.10490666>.
