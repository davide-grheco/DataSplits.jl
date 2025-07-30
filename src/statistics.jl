using Statistics
using MLUtils: numobs, getobs

"""
    pooled_covariance(groups::Vector{<:AbstractMatrix}) -> Matrix{Float64}

Compute the pooled covariance matrix from a list of data matrices (`groups`), assuming all groups share the same true covariance.

Each matrix in `groups` is one group (rows = features, columns = samples).
"""
function pooled_covariance(groups::Vector{<:AbstractMatrix})::Matrix{Float64}
  total_weight = 0
  n_features = length(getobs(groups[1], 1))
  pooled_cov = zeros(Float64, n_features, n_features)

  for group in groups
    n = numobs(group)
    n <= 1 && continue
    cov_group = cov(group; dims = 2, corrected = true)
    weight = n - 1
    pooled_cov .+= weight * cov_group
    total_weight += weight
  end

  total_weight > 0 || throw(ArgumentError("Total weight must be > 0"))
  return pooled_cov / total_weight
end
