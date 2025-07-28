using Distances
using MLUtils: numobs, getobs

"""
    find_maximin_element(distances::AbstractMatrix{T},
                        source_set::AbstractVector{Int},
                        reference_set::AbstractVector{Int}) -> Int

Find the element in `source_set` that maximizes the minimum distance to all elements in `reference_set`.

# Arguments
- `distances::AbstractMatrix{T}`: Precomputed, symmetric pairwise distance matrix.
- `source_set::AbstractVector{Int}`: Set of items to evaluate.
- `reference_set::AbstractVector{Int}`: Set of items to compare against.

# Returns
- `Int`: The index in `source_set` that is **farthest from its nearest neighbour** in `reference_set`.

# Notes
- If `reference_set` is empty, throws `ArgumentError`
- Breaks ties by returning the first maximum
"""
function find_maximin_element(
  distances::AbstractMatrix{T},
  source_set::Union{AbstractVector{Int},AbstractSet{Int}},
  reference_set::Union{AbstractVector{Int},AbstractSet{Int}},
) where {T<:Real}

  isempty(source_set) && throw(ArgumentError("source_set cannot be empty"))
  isempty(reference_set) && throw(ArgumentError("reference_set cannot be empty"))

  best_index = first(source_set)
  best_score = -Inf

  for i in source_set
    min_dist = minimum(distances[i, j] for j in reference_set)
    if min_dist > best_score
      best_score = min_dist
      best_index = i
    end
  end

  return best_index
end

"""
    distance_matrix(X, metric::PreMetric)

Compute the full symmetric pairwise distance matrix `D` for the dataset `X`
using the given `metric`. This function uses `get_sample` to access samples,
ensuring compatibility with any custom array type that implements
`get_sample` and `sample_indices`.

Returns a matrix `D` such that `D[i, j] = metric(xᵢ, xⱼ)` and `D[i, j] == D[j, i]`.
"""
function distance_matrix(X, metric::PreMetric)
  N = numobs(X)
  D = zeros(Float64, N, N)

  for i = 1:N-1
    xi = getobs(X, i)
    for j = i+1:N
      d = evaluate(metric, xi, getobs(X, j))
      D[i, j] = D[j, i] = d
    end
  end

  return D
end

function distance_matrix(X::AbstractMatrix, metric::PreMetric)
  return pairwise(metric, X, X; dims = 2)
end
