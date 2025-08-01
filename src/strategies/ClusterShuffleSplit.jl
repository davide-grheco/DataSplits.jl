using Random
import Clustering: ClusteringResult, assignments
using Clustering

"""
    ClusterShuffleSplit(res::ClusteringResult, frac::Real)
    ClusterShuffleSplit(f::Function, frac::Real, data)

Group-aware train/test splitting strategy that accumulates whole clusters in the training set until the requested fraction is reached.

# Fields
- `clusters::ClusteringResult`: Clustering assignments for the data.
- `frac::ValidFraction{T}`: Fraction of samples in the training set (0 < frac < 1).

# Notes

The clusters are accorpated to the training set in a random fashion. As such it is possible to overshoot the requested fraction. This algorithm does not attempt to obtain the split most similar to the requested fraction possible.

# Examples
```julia
splitter = ClusterShuffleSplit(clustering_result, 0.7)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```
"""
struct ClusterShuffleSplit{T<:Real} <: SplitStrategy
  clusters::ClusteringResult
  frac::ValidFraction{T}
end

# Precomputed clusters
ClusterShuffleSplit(res::ClusteringResult, frac::Real) =
  ClusterShuffleSplit(res, ValidFraction(frac))

# On-the-fly clustering
ClusterShuffleSplit(f::Function, frac::Real, data) = ClusterShuffleSplit(f(data), frac)

function _split(data, s::ClusterShuffleSplit; rng = Random.default_rng())
  N = numobs(data)
  assigns = assignments(s.clusters)
  cl_ids = unique(assigns)
  shuffle!(rng, cl_ids)
  train_pos = Int[]
  train_pos = Int[]
  test_pos = Int[]
  for cid in cl_ids
    if length(train_pos) / N < float(s.frac)
      cluster_indices = findall(==(cid), assigns)
      if (length(train_pos) + length(cluster_indices)) / N <= float(s.frac)
        append!(train_pos, cluster_indices)
      else
        # Split this cluster to hit the fraction as closely as possible
        n_needed = round(Int, float(s.frac) * N) - length(train_pos)
        if n_needed > 0
          append!(train_pos, cluster_indices[1:n_needed])
          append!(test_pos, cluster_indices[n_needed+1:end])
        else
          append!(test_pos, cluster_indices)
        end
        break
      end
    else
      break
    end
  end
  # Add any remaining clusters to test
  for cid in cl_ids
    cluster_indices = findall(==(cid), assigns)
    for idx in cluster_indices
      if !(idx in train_pos) && !(idx in test_pos)
        push!(test_pos, idx)
      end
    end
  end
  train_pos = unique(train_pos)
  test_pos = unique(test_pos)
  return TrainTestSplit(train_pos, test_pos)
end
