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
function _split(data, ::First80Split; rng=nothing)
  N = length(sample_indices(data))
  cut = floor(Int, 0.8 * N)
  idxs = collect(sample_indices(data))
  train = idxs[1:cut]
  test = idxs[cut+1:end]
  return train, test
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
