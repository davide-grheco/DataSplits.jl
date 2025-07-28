```@meta
CurrentModule = DataSplits
```

# OptiSim Split

OptiSim is an iterative, optimisable splitting algorithm that aims to maximize the diversity of the training set by minimizing within-set similarity. At each step, it generates a temporary subsample of candidates and selects the one most dissimilar to the current training set, repeating until the desired fraction is reached. The algorithm is flexible and can be tuned for speed or quality by adjusting the subsample size and distance cutoff.

## How it works

1. Compute the pairwise distance matrix for all samples.
2. Start with a random sample in the training set.
3. At each iteration, generate a candidate subsample.
4. Select the candidate most dissimilar to the current training set.
5. Repeat until the desired number of training samples is reached.

## Usage

```julia
using DataSplits
train, test = split(X, OptiSimSplit(0.8; max_subsample_size=50))
```

- `X`: Data matrix
- `0.8`: Fraction of samples to use for training
- `max_subsample_size`: Size of candidate pool at each step (default: 0, i.e., use all)
- `distance_cutoff`: Threshold for similarity filtering (default: 0.35)
- `metric`: Distance metric (default: Euclidean)

## Options

- `OptiSimSplit(frac; max_subsample_size=0, distance_cutoff=0.35, metric=Euclidean())`

## Notes

- This algorithm depends on the first selection which is random. Different runs may result in very different compositions of training and test set.
- **MinimumDissimilaritySplit** is an alias for `OptiSimSplit` with `max_subsample_size=1`. This provides a fast, greedy variant for large datasets where speed is more important than optimal diversity.
- **MaximumDissimilaritySplit** is an alias for `OptiSimSplit` with `max_subsample_size=N`. This yields a greedy, diversity-maximizing split. Note: This implementation does **not** discard the first two samples as in the original Maximum Dissimilarity algorithm, but otherwise follows the same greedy logic. This algorithm is known to greedily include outliers.

---

## Reference

Clark, R. D. OptiSim:  An Extended Dissimilarity Selection Method for Finding Diverse Representative Subsets. J. Chem. Inf. Comput. Sci. 1997, 37 (6), 1181–1188. <https://doi.org/10.1021/ci970282v>.
