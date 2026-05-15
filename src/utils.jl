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

function find_maximin_element_lazy(X, metric, source_set, reference_set)
  best_index = first(source_set)
  best_score = -Inf
  for i in source_set
    value = getobs(X, i)
    min_dist = minimum(j -> Distances.evaluate(metric, value, getobs(X, j)), reference_set)
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

  for i = 1:(N-1)
    xi = getobs(X, i)
    for j = (i+1):N
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
    groupsortperm(v) -> (sorted_keys, perm)

Return the sorted unique values of `v` and a stable sort permutation of `v`.

`perm` is a permutation of `1:length(v)` such that `v[perm]` is non-decreasing.
`sorted_keys == unique(v[perm])`. Together, `sorted_keys` and `perm` partition
every index in `1:length(v)` with no duplicates.
"""
function groupsortperm(v)
  perm = sortperm(v, alg = MergeSort)
  return unique(view(v, perm)), perm
end

"""
    group_offsets(sorted_keys, perm, v) -> block_offset

Compute block-boundary offsets for a grouped, sorted permutation.

`block_offset[b]+1 : block_offset[b+1]` are the positions in `perm` (equivalently
the slice of `v[perm]`) whose value equals `sorted_keys[b]`.
`block_offset[1] == 0` and `block_offset[end] == length(perm)`.
"""
function group_offsets(sorted_keys, perm, v)
  sorted_v = view(v, perm)
  B = length(sorted_keys)
  block_offset = Vector{Int}(undef, B + 1)
  block_offset[1] = 0
  for (b, k) in enumerate(sorted_keys)
    block_offset[b+1] = searchsortedlast(sorted_v, k)
  end
  return block_offset
end

"""
    distribute_blocks(B::Int, n_chunks::Int) -> chunk_block_end

Distribute `B` contiguous blocks across `n_chunks` as evenly as possible
(matching `numpy.array_split` semantics: the remainder is spread over the
first `B mod n_chunks` chunks). Returns a vector of length `n_chunks` where
`chunk_block_end[c]` is the index of the last block in chunk `c`.
Chunk sizes differ by at most 1.
"""
function distribute_blocks(B::Int, n_chunks::Int)
  base, rem = divrem(B, n_chunks)
  chunk_block_end = Vector{Int}(undef, n_chunks)
  acc = 0
  for c = 1:n_chunks
    acc += base + (c <= rem ? 1 : 0)
    chunk_block_end[c] = acc
  end
  return chunk_block_end
end

