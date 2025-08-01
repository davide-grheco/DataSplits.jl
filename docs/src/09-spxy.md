```@meta
CurrentModule = DataSplits
```

# SPXY Split

## Overview

The SPXY algorithm extends Kennard–Stone by considering both the feature matrix (`X`) and the target vector (`y`) when splitting data. It constructs a joint distance matrix as the sum of normalized pairwise distances in `X` and `y`, then applies the maximin selection. This ensures the training set is diverse in both predictors and response, which is especially important for regression tasks where the target distribution matters.

`MDKSSplit(frac)`: Alias for `SPXYSplit(frac; metric=Mahalanobis())` (SPXY algorithm using Mahalanobis distance).

## How it works

1. Compute the normalized pairwise distance matrix for `X` (features).
2. Compute the normalized pairwise distance matrix for `y` (target).
3. Add the two matrices to form a joint distance matrix.
4. Apply the Kennard–Stone maximin selection on the joint distance matrix to select a representative training set.

## Usage

```julia
using DataSplits, Distances
splitter = SPXYSplit(0.7; metric=Cityblock())
result = split((X, y), splitter)
X_train, X_test = splitdata(result, X)
```

## Notes/Limitations

- Most appropriate for regression and continuous targets
- You must call `split((X, y), strategy)` or `split(X, y, strategy)`; calling `split(X, strategy)` will error

## API Reference

- [`SPXYSplit`](@ref)
- [`MDKSSplit`](@ref)
- [`split`](@ref)
- [`splitdata`](@ref)

## References

Galvao, R.; Araujo, M.; Jose, G.; Pontes, M.; Silva, E.; Saldanha, T. A Method for Calibration and Validation Subset Partitioning. Talanta 2005, 67 (4), 736–740. <https://doi.org/10.1016/j.talanta.2005.03.025>.
Saptoro, A.; Tadé, M. O.; Vuthaluru, H. A Modified Kennard-Stone Algorithm for Optimal Division of Data for Developing Artificial Neural Network Models. Chemical Product and Process Modeling 2012, 7 (1). <https://doi.org/10.1515/1934-2659.1645>.
