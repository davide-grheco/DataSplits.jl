```@meta
CurrentModule = DataSplits
```

# 06. Examples & Tutorials

## Example: SPXY Split on Custom Data

```julia
using DataSplits, Distances
# Suppose `df` is a DataFrame with features and target
df = DataFrame(rand(100, 5), :auto)
y = rand(100)
train, test = split((Matrix(df), y), SPXYSplit(0.75))
```

## Example: Custom Splitter Implementation

Suppose you want to create a splitter that always assigns the first 80% of samples to train:

```julia
struct First80Split <: SplitStrategy end

function first80(N, s, rng, data)
    cut = floor(Int, 0.8 * N)
    train_pos = 1:cut
    test_pos = (cut+1):N
    return train_pos, test_pos
end

function _split(data, ::First80Split; rng=nothing)
    split_with_positions(data, nothing, first80; rng=rng)
end
```

## Example: Group-aware Splitting

```julia
using DataSplits
# Suppose you have cluster assignments for your data
clusters = [rand(1:5) for _ in 1:100]
using Clustering
res = Clustering.ClusteringResult(clusters)
train, test = split(X, ClusterShuffleSplit(res, 0.7))
```
