```@meta
CurrentModule = DataSplits
```

# 06. Examples & Tutorials

**Note:** DataSplits expects data matrices to be in the Julia ML convention: columns are samples, rows are features. If your data uses rows as samples, transpose it before splitting (e.g., use `X'`).

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

function _split(data, ::First80Split; rng=nothing)
    N = numobs(data)
    cut = floor(Int, 0.8 * N)
    train_pos = 1:cut
    test_pos = (cut+1):N
    return TrainTestSplit(train_pos, test_pos)
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
