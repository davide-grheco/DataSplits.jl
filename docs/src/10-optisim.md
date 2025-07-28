```@meta
CurrentModule = DataSplits
```

# OptiSim Split

## Overview

OptiSim is an iterative, optimisable splitting algorithm that aims to maximize the diversity of the training set by minimizing within-set similarity. At each step, it generates a temporary subsample of candidates and selects the one most dissimilar to the current training set, repeating until the desired fraction is reached. The algorithm is flexible and can be tuned for speed or quality by adjusting the subsample size and distance cutoff.

## How it works

1. Compute the pairwise distance matrix for all samples.
2. Start with a random sample in the training set.
3. At each iteration, generate a candidate subsample.
4. Select the candidate most dissimilar to the current training set.
5. Repeat until the desired number of training samples is reached.

## Arguments

- `frac`: Fraction of samples to use for training (0 < frac < 1)
- `max_subsample_size`: Size of candidate pool at each step (default: 0, i.e., use all)
- `distance_cutoff`: Threshold for similarity filtering (default: 0.35)
- `metric`: Distance metric (default: Euclidean)

## Usage

```julia
using DataSplits
splitter = OptiSimSplit(0.8; max_subsample_size=50)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```

## Notes/Limitations

- Depends on the first selection, which is random
- **MinimumDissimilaritySplit** is an alias for `OptiSimSplit` with `max_subsample_size=1`. This provides a fast, greedy variant for large datasets where speed is more important than optimal diversity.
- **MaximumDissimilaritySplit** is an alias for `OptiSimSplit` with `max_subsample_size=N`. This yields a greedy, diversity-maximizing split. Note: This implementation does **not** discard the first two samples as in the original Maximum Dissimilarity algorithm, but otherwise follows the same greedy logic. This algorithm is known to greedily include outliers.

## API Reference

- [`OptiSimSplit`](@ref)
- [`split`](@ref)
- [`splitdata`](@ref)

## References

Clark, R. D. OptiSim:  An Extended Dissimilarity Selection Method for Finding Diverse Representative Subsets. J. Chem. Inf. Comput. Sci. 1997, 37 (6), 1181–1188. <https://doi.org/10.1021/ci970282v>.
