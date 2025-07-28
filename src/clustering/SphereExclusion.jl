using Distances
import Clustering: ClusteringResult, assignments, counts, nclusters, wcounts
import DataSplits: distance_matrix
using StatsBase: minmax

"""
    SphereExclusionResult

Result of sphere exclusion clustering.

Fields:
- `assignments::Vector{Int}`: cluster index per point.
- `radius::Float64`: exclusion radius.
- `metric::Distances.SemiMetric`: distance metric.
"""
struct SphereExclusionResult <: ClusteringResult
  assignments::Vector{Int}
  radius::Float64
  metric::Distances.SemiMetric
end

assignments(R::SphereExclusionResult) = R.assignments
nclusters(R::SphereExclusionResult) = isempty(R.assignments) ? 0 : maximum(R.assignments)
counts(R::SphereExclusionResult) = [count(==(k), R.assignments) for k = 1:nclusters(R)]
wcounts(R::SphereExclusionResult) = Float64.(counts(R))

"""
    sphere_exclusion(data; radius::Real, metric::Distances.SemiMetric=Euclidean()) -> SphereExclusionResult

Cluster samples in `data` by sphere exclusion:
1. Compute full pairwise distance matrix and normalize values to [0,1].
2. While unassigned samples remain:
   - Pick first unassigned sample `i`.
   - All unassigned samples `j` with normalized distance `D[i,j] <= radius` form a cluster.
   - Mark them assigned and increment cluster ID.
"""
function sphere_exclusion(data; radius::Real, metric::Distances.SemiMetric = Euclidean())
  N = numobs(data)
  if N == 0
    return SphereExclusionResult(Int[], float(radius), metric)
  end
  D = distance_matrix(data, metric)
  mn = minimum(D)
  mx = maximum(D)
  if mx > mn
    D .= (D .- mn) ./ (mx - mn)
  else
    fill!(D, zero(eltype(D)))
  end
  un = Set(1:N)
  assign = zeros(Int, N)
  cid = 1
  rad = float(radius)
  while !isempty(un)
    i = first(un)
    members = [j for j in un if D[i, j] <= rad]
    for j in members
      assign[j] = cid
      delete!(un, j)
    end
    cid += 1
  end
  SphereExclusionResult(assign, rad, metric)
end
