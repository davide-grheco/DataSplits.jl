```@meta
CurrentModule = DataSplits
```

# Examples & Tutorials

**Note:** DataSplits expects data matrices to be in the Julia ML convention: columns are samples, rows are features. If your data uses rows as samples, transpose it before splitting (e.g., use `X'`).

## Example: SPXY Split on Custom Data

```julia
using DataSplits, Distances
# Suppose `df` is a DataFrame with features and target
df = DataFrame(rand(100, 5), :auto)
y = rand(100)
splitter = SPXYSplit(0.75)
result = split((Matrix(df), y), splitter)
X_train, X_test = splitdata(result, Matrix(df))
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
using DataSplits, Clustering
clusters = sphere_exclusion(X; radius=0.3)
splitter = ClusterShuffleSplit(clusters, 0.7)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```
