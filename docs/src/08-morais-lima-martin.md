```@meta
CurrentModule = DataSplits
```

# Morais–Lima–Martin Split

## Overview

The Morais–Lima–Martin algorithm performs a Kennard–Stone split, then randomly swaps a fraction of samples between the training and test sets. This introduces additional randomness to balance representativeness and variability.

## How it works

1. Apply Kennard–Stone to select the initial training set.
2. Randomly select a fraction of samples from both train and test sets.
3. Swap these selected samples between the two sets.

## Arguments

- `frac`: Fraction of samples to use for training (0 < frac < 1)
- `swap_frac`: Fraction of samples to swap between train and test (0 < swap_frac < 1)
- `metric`: Distance metric to use for Kennard–Stone (default: Euclidean)

## Usage

```julia
using DataSplits, Distances
splitter = MoraisLimaMartinSplit(0.8; swap_frac=0.1, metric=Cityblock())
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```

## Reference

Morais, C. L. M.; Santos, M. C. D.; Lima, K. M. G.; Martin, F. L. Improving Data Splitting for Classification Applications in Spectrochemical Analyses Employing a Random-Mutation Kennard-Stone Algorithm Approach. Bioinformatics 2019, 35 (24), 5257–5263. <https://doi.org/10.1093/bioinformatics/btz421>.

## API Reference

- [`MoraisLimaMartinSplit`](@ref)
- [`split`](@ref)
- [`splitdata`](@ref)
