using Distances
import Clustering: ClusteringResult, assignments, counts, nclusters, wcounts
import DataSplits: distance_matrix
using StatsBase: minmax

"""
    SphereExclusionResult

Result of sphere exclusion clustering.

# Fields
- `assignments::Vector{Int}`: Cluster index per point (1-based).
- `radius::Float64`: Exclusion radius used for clustering.
- `metric::Distances.SemiMetric`: Distance metric used.
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

Clusters samples in `data` using the sphere exclusion algorithm.

# Arguments
- `data`: Data matrix or container. Columns are samples.
- `radius::Real`: Exclusion radius (normalized to [0, 1]).
- `metric::Distances.SemiMetric`: Distance metric (default: Euclidean()).

# Returns
- `SphereExclusionResult`: Clustering result with assignments, radius, and metric.

# Notes
- The distance matrix is normalized to [0, 1] before clustering.
- Each cluster contains all points within `radius` of the cluster center.

# Examples
```julia
result = sphere_exclusion(X; radius=0.2)
assignments = result.assignments
```
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
