using Random
using Statistics: std, mean
using Clustering

"""
    ClusterStratifiedSplit(res::ClusteringResult, allocation::Symbol; n=nothing, frac)
    ClusterStratifiedSplit(f::Function, allocation::Symbol; n=nothing, frac, data)

Cluster-stratified train/test splitting strategy with flexible allocation methods.

# Fields
- `clusters::ClusteringResult`: Clustering assignments for the data.
- `allocation::Symbol`: Allocation method (`:equal`, `:proportional`, or `:neyman`).
- `n::Union{Nothing,Int}`: Number of samples per cluster (for `:equal`/`:neyman`).
- `frac::Real`: Fraction of selected samples to use for training (rest go to test).

# Examples
```julia
splitter = ClusterStratifiedSplit(clustering_result, :proportional; frac=0.8)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)

# Reference

For a complete analysis of several methodologies refer to:

May, R. J.; Maier, H. R.; Dandy, G. C. Data Splitting for Artificial Neural Networks Using SOM-Based Stratified Sampling. Neural Networks 2010, 23 (2), 283–294. https://doi.org/10.1016/j.neunet.2009.11.009.

The implementation of equal is different than the one introduced in the paper and in previous references.
```
"""
struct ClusterStratifiedSplit <: SplitStrategy
  clusters::ClusteringResult
  allocation::Symbol
  n::Union{Nothing,Int}
  frac::Real
end

ClusterStratifiedSplit(res::ClusteringResult, allocation::Symbol; n = nothing, frac) =
  ClusterStratifiedSplit(res, allocation, n, frac)

ClusterStratifiedSplit(f::Function, allocation::Symbol; n = nothing, frac, data) =
  ClusterStratifiedSplit(f(data), allocation; n = n, frac = frac)

"""
    equal_allocation(cl_ids, idxs_by_cluster, n, rng)

Randomly select `n` samples from each cluster (or all if cluster is smaller).
Returns a Dict mapping cluster id to selected indices.
"""
function equal_allocation(cl_ids, idxs_by_cluster, n, rng)
  if isnothing(n)
    throw(SplitParameterError("Parameter n must be provided for equal allocation."))
  end
  selected = Dict{Int,Vector{Int}}()
  for cid in cl_ids
    idxs = idxs_by_cluster[cid]
    n_select = min(n, length(idxs))
    idxs_shuffled = copy(idxs)
    shuffle!(rng, idxs_shuffled)
    selected[cid] = idxs_shuffled[1:n_select]
  end
  return selected
end

"""
    proportional_allocation(cl_ids, idxs_by_cluster, rng)

Use all samples in each cluster, shuffled.
Returns a Dict mapping cluster id to selected indices.
"""
function proportional_allocation(cl_ids, idxs_by_cluster, rng)
  selected = Dict{Int,Vector{Int}}()
  for cid in cl_ids
    idxs = idxs_by_cluster[cid]
    idxs_shuffled = copy(idxs)
    shuffle!(rng, idxs_shuffled)
    selected[cid] = idxs_shuffled
  end
  return selected
end

"""
    neyman_allocation(cl_ids, idxs_by_cluster, n, data, rng)

Randomly select a Neyman quota from each cluster (proportional to cluster size and mean std of features).
Returns a Dict mapping cluster id to selected indices.
"""
function neyman_allocation(cl_ids, idxs_by_cluster, n, data, rng)
  if isnothing(n)
    throw(SplitParameterError("Parameter n must be provided for Neyman allocation."))
  end
  stds = Dict{Int,Float64}()
  for cid in cl_ids
    idxs = idxs_by_cluster[cid]
    cluster_data = data[idxs, :]
    stds[cid] = mean(std(cluster_data, dims = 1))
  end
  numerators = Dict(cid => length(idxs_by_cluster[cid]) * stds[cid] for cid in cl_ids)
  denom = sum(values(numerators))
  total = n * length(cl_ids)
  selected = Dict{Int,Vector{Int}}()
  for cid in cl_ids
    prop = numerators[cid] / denom
    n_select = max(1, round(Int, prop * total))
    idxs = idxs_by_cluster[cid]
    idxs_shuffled = copy(idxs)
    shuffle!(rng, idxs_shuffled)
    selected[cid] = idxs_shuffled[1:min(n_select, length(idxs))]
  end
  return selected
end

"""
    cluster_stratified(N, s, rng, data)

Main splitting function. For each cluster, selects indices according to the allocation method,
then splits those indices into train/test according to `frac`.
"""
function _split(data, s::ClusterStratifiedSplit; rng = Random.default_rng())
  N = numobs(data)
  assigns = assignments(s.clusters)
  cl_ids = unique(assigns)
  idxs_by_cluster = Dict(cid => findall(==(cid), assigns) for cid in cl_ids)
  selected = if s.allocation == :equal
    equal_allocation(cl_ids, idxs_by_cluster, s.n, rng)
  elseif s.allocation == :proportional
    proportional_allocation(cl_ids, idxs_by_cluster, rng)
  elseif s.allocation == :neyman
    neyman_allocation(cl_ids, idxs_by_cluster, s.n, data, rng)
  else
    throw(
      SplitParameterError(
        "Unknown allocation method: $(s.allocation). Please use one of :equal, :proportional, or :neyman.",
      ),
    )
  end
  train_pos = Int[]
  test_pos = Int[]
  for cid in cl_ids
    idxs = selected[cid]
    n_train = ceil(Int, s.frac * length(idxs))
    n_train = min(n_train, length(idxs))
    train_idxs = idxs[1:n_train]
    test_idxs = idxs[n_train+1:end]
    append!(train_pos, train_idxs)
    append!(test_pos, test_idxs)
  end
  return TrainTestSplit(train_pos, test_pos)
end
