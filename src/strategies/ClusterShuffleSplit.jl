using Random
import Clustering: ClusteringResult, assignments
using Clustering

"""
    ClusterShuffleSplit(res::ClusteringResult, frac::Real)
    ClusterShuffleSplit(f::Function, frac::Real, data; rng)

Group-aware train/test splitter: accepts either:

1. A precomputed ClusteringResult.
2. A clustering function f(data) that returns one.

At construction, clustering is executed so the strategy always holds a ClusteringResult.

Arguments:
- `res` or `f(...)
- `frac`: fraction of samples in the training set (0 < frac < 1).

This splitter shuffles cluster IDs and accumulates whole clusters until
`frac * N` samples are in the train set, then returns `(train_idx, test_idx)`.
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

function cluster_shuffle(N, s, rng, data)
  assigns = assignments(s.clusters)
  cl_ids = unique(assigns)
  shuffle!(rng, cl_ids)
  train_pos = Int[]
  for cid in cl_ids
    if length(train_pos) / N < float(s.frac)
      append!(train_pos, findall(==(cid), assigns))
    else
      break
    end
  end
  train_pos = unique(train_pos)
  test_pos = setdiff(1:N, train_pos)
  return TrainTestSplit(train_pos, test_pos)
end

function _split(data, s::ClusterShuffleSplit; rng = Random.default_rng())
  split_with_positions(data, s, cluster_shuffle; rng = rng)
end
