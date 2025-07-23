# Cluster Stratified Split

Cluster Stratified Split is a train/test splitting strategy that ensures each cluster (as determined by a clustering algorithm) is split into train and test sets according to a specified allocation method and user-defined train fraction.

## Allocation Methods

- **Equal allocation**: Randomly selects a fixed number `n` of samples from each cluster, then splits those into train/test according to the user-specified `frac` (fraction for train set). The rest of the cluster is unused.
- **Proportional allocation**: Uses all samples in each cluster, splits them into train/test according to the user-specified `frac` (fraction for train set).
- **Neyman allocation**: Randomly selects a quota from each cluster (proportional to cluster size and mean standard deviation of features), then splits those into train/test according to the user-specified `frac` (fraction for train set). The rest of the cluster is unused.

## Usage

```julia
using DataSplits, Clustering

# Assume you have a ClusteringResult `clusters` and a data matrix X
splitter = ClusterStratifiedSplit(clusters, :equal; n=4, frac=0.7)
train_idx, test_idx = split(X, splitter)

splitter = ClusterStratifiedSplit(clusters, :proportional; frac=0.7)
train_idx, test_idx = split(X, splitter)

splitter = ClusterStratifiedSplit(clusters, :neyman; n=4, frac=0.7)
train_idx, test_idx = split(X, splitter)
```

## Arguments

- `clusters`: ClusteringResult (from Clustering.jl)
- `allocation`: `:equal`, `:proportional`, or `:neyman`
- `n`: Number of samples per cluster (for `:equal` and `:neyman`)
- `frac`: Fraction of selected samples to use for train (rest go to test)

## Notes

- If `n` is greater than the cluster size, all samples in the cluster are used.
- For `:proportional`, all samples are always used.
- For `frac=1.0`, all selected samples go to train; for `frac=0.0`, all go to test.
- The split is performed **per cluster**; rounding is handled so that the train set always gets the larger share when the split is not exact.

See also: [Clustering.jl](https://github.com/JuliaStats/Clustering.jl)
