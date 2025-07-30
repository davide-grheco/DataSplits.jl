using Distances
using MLUtils: numobs, getobs

"""
    find_maximin_element(distances::AbstractMatrix{T},
                        source_set::Union{AbstractVector{Int},AbstractSet{Int}},
                        reference_set::Union{AbstractVector{Int},AbstractSet{Int}}) -> Int

Finds the element in `source_set` that maximizes the minimum distance to all elements in `reference_set`.

# Arguments
- `distances::AbstractMatrix{T}`: Precomputed, symmetric pairwise distance matrix (N×N).
- `source_set::Union{AbstractVector{Int},AbstractSet{Int}}`: Indices to evaluate.
- `reference_set::Union{AbstractVector{Int},AbstractSet{Int}}`: Indices to compare against.

# Returns
- `Int`: Index in `source_set` that is farthest from its nearest neighbor in `reference_set`.

# Notes
- Throws `ArgumentError` if `reference_set` is empty.
- Breaks ties by returning the first maximum.
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

Computes the full symmetric pairwise distance matrix for the dataset `X` using the given `metric`.

# Arguments
- `X`: Data matrix or container. Columns are samples (features × samples).
- `metric::PreMetric`: Distance metric from Distances.jl.

# Returns
- `D::Matrix{Float64}`: Matrix where `D[i, j] = metric(xᵢ, xⱼ)` and `D[i, j] == D[j, i]`.

# Notes
- For custom containers, `getobs(X, i)` is used to access samples.
- The matrix is symmetric and not normalized.
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

"""
    SplitInputError(msg)

Error thrown when input data to a split is invalid (e.g., empty, wrong shape, mismatched X/y).
"""
struct SplitInputError <: Exception
  msg::String
end

"""
    SplitParameterError(msg)

Error thrown when split parameters are invalid (e.g., unknown allocation, out-of-bounds fraction).
"""
struct SplitParameterError <: Exception
  msg::String
end

"""
    SplitNotImplementedError(msg)

Error thrown when a required split method or feature is not implemented.
"""
struct SplitNotImplementedError <: Exception
  msg::String
end



"""
    train_test_counts(N, frac; min_train=2, min_test=2)

Given total sample count `N` and train fraction `frac`, return `(n_train, n_test)`.
Throws `SplitParameterError` if the split is not possible (e.g., too few samples, fraction out of bounds).
"""
function train_test_counts(N::Integer, frac; min_train::Integer = 1, min_test::Integer = 1)
  if N < min_train + min_test
    throw(
      SplitParameterError(
        "Not enough samples ($N) to split: need at least $(min_train + min_test).",
      ),
    )
  end
  n_train = round(Int, frac * N)
  n_test = N - n_train

  if n_train < min_train || n_test < min_test
    throw(
      SplitParameterError(
        "Split would result in too few samples: n_train=$n_train, n_test=$n_test (min_train=$min_train, min_test=$min_test).",
      ),
    )
  end
  return n_train, n_test
end
