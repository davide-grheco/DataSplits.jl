using Random
using Statistics: std, mean

"""
    GroupStratifiedSplit(allocation::Symbol; n=nothing) <: AbstractSplitStrategy

Group-stratified train/test splitting with flexible allocation methods.

Groups are passed as a vector of membership IDs via the `groups=` keyword.
Any grouping is valid: cluster assignments, patient IDs, scaffold labels,
batch numbers, site identifiers, graph communities, etc.

# Fields
- `allocation::Symbol`: Allocation method — `:equal`, `:proportional`, or `:neyman`.
- `n::Union{Nothing,Int}`: Samples per group for `:equal` and `:neyman`.

# Allocation methods
- `:proportional` — use all samples from each group (shuffled).
- `:equal` — select `n` samples from each group (requires `n`).
- `:neyman` — select proportional to group size × within-group std (requires `n`).

The training fraction within each group is derived from the global cohort
sizes (`n_train / N`).

# Examples
```julia
res = partition(X, GroupStratifiedSplit(:proportional);
                groups=patient_ids, train=80, test=20)
X_train, X_test = splitdata(res, X)

# With Clustering.jl
using Clustering
res = partition(X, GroupStratifiedSplit(:equal; n=5);
                groups=assignments(kmeans(X, 4)), train=80, test=20)
```

# References
May, R. J.; Maier, H. R.; Dandy, G. C. Data Splitting for Artificial Neural Networks
Using SOM-Based Stratified Sampling. *Neural Networks* 2010, 23(2), 283–294.
"""
struct GroupStratifiedSplit <: AbstractSplitStrategy
  allocation::Symbol
  n::Union{Nothing,Int}
end

GroupStratifiedSplit(allocation::Symbol; n = nothing) = GroupStratifiedSplit(allocation, n)

consumes(::GroupStratifiedSplit) = (:groups,)
fallback_from_data(::GroupStratifiedSplit) = (:groups,)

function _equal_allocation(cl_ids, idxs_by_cluster, n, rng)
  isnothing(n) &&
    throw(SplitParameterError("Parameter n must be provided for equal allocation."))
  selected = Dict{Any,Vector{Int}}()
  for cid in cl_ids
    idxs = shuffle(rng, idxs_by_cluster[cid])
    selected[cid] = idxs[1:min(n, length(idxs))]
  end
  return selected
end

function _proportional_allocation(cl_ids, idxs_by_cluster, rng)
  selected = Dict{Any,Vector{Int}}()
  for cid in cl_ids
    selected[cid] = shuffle(rng, idxs_by_cluster[cid])
  end
  return selected
end

function _neyman_allocation(cl_ids, idxs_by_cluster, n, data, rng)
  isnothing(n) &&
    throw(SplitParameterError("Parameter n must be provided for Neyman allocation."))
  stds = Dict{Any,Float64}()
  for cid in cl_ids
    cluster_data = getobs(data, idxs_by_cluster[cid])
    stds[cid] = mean(std(cluster_data, dims = 2))
  end
  numerators = Dict(cid => length(idxs_by_cluster[cid]) * stds[cid] for cid in cl_ids)
  denom = sum(values(numerators))
  total = n * length(cl_ids)
  selected = Dict{Any,Vector{Int}}()
  for cid in cl_ids
    prop = numerators[cid] / denom
    n_select = max(1, round(Int, prop * total))
    idxs = shuffle(rng, idxs_by_cluster[cid])
    selected[cid] = idxs[1:min(n_select, length(idxs))]
  end
  return selected
end

function _partition(
  data,
  s::GroupStratifiedSplit;
  groups,
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  N = numobs(data)
  frac = n_train / N
  cl_ids = unique(groups)
  idxs_by_cluster = Dict(cid => findall(==(cid), groups) for cid in cl_ids)
  selected = if s.allocation == :equal
    _equal_allocation(cl_ids, idxs_by_cluster, s.n, rng)
  elseif s.allocation == :proportional
    _proportional_allocation(cl_ids, idxs_by_cluster, rng)
  elseif s.allocation == :neyman
    _neyman_allocation(cl_ids, idxs_by_cluster, s.n, data, rng)
  else
    throw(
      SplitParameterError(
        "Unknown allocation method: $(s.allocation). Use :equal, :proportional, or :neyman.",
      ),
    )
  end
  train_pos = Int[]
  test_pos = Int[]
  for cid in cl_ids
    idxs = selected[cid]
    n_train_cid = min(ceil(Int, frac * length(idxs)), length(idxs))
    append!(train_pos, idxs[1:n_train_cid])
    append!(test_pos, idxs[(n_train_cid+1):end])
  end
  return TrainTestSplit(train_pos, test_pos)
end
