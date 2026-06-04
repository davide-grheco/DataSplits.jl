using Distances
using MLUtils: numobs, getobs, obsview

_obs(data::AbstractMatrix, i) = obsview(data, i)
_obs(data, i) = getobs(data, i)

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
    value = _obs(X, i)
    min_dist = minimum(j -> Distances.evaluate(metric, value, _obs(X, j)), reference_set)
    if min_dist > best_score
      best_score = min_dist
      best_index = i
    end
  end
  return best_index
end

"""
    distance_matrix(X, metric::Distances.SemiMetric)

Computes the full symmetric pairwise distance matrix for the dataset `X` using the given `metric`.

# Arguments
- `X`: Data matrix or container. Columns are samples (features × samples).
- `metric::Distances.SemiMetric`: Distance metric from Distances.jl.

# Returns
- `D::Matrix{Float64}`: Matrix where `D[i, j] = metric(xᵢ, xⱼ)` and `D[i, j] == D[j, i]`.

# Notes
- For custom containers, `getobs(X, i)` is used to access samples.
- The matrix is symmetric and not normalized.
"""
function distance_matrix(X, metric::Distances.SemiMetric)
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

function distance_matrix(X::AbstractMatrix, metric::Distances.SemiMetric)
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

Base.showerror(io::IO, e::SplitInputError) = print(io, "SplitInputError: ", e.msg)
Base.showerror(io::IO, e::SplitParameterError) = print(io, "SplitParameterError: ", e.msg)
Base.showerror(io::IO, e::SplitNotImplementedError) =
  print(io, "SplitNotImplementedError: ", e.msg)

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

"""
    _warn_undershoot(n_selected, n_requested, msg; id)

Emit a `@warn` when fewer samples were selected than requested.
`id` is used as the log record `_id` for selective filtering with
`LoggingExtras.EarlyFilteredLogger`.
"""
function _warn_undershoot(
  n_selected::Int,
  n_requested::Int,
  msg::String;
  id::Symbol = :datasplits_undershoot,
)
  n_selected < n_requested || return
  @warn msg _id = id _group = :datasplits
end

_squared_metric(::Euclidean) = SqEuclidean()
_squared_metric(m::Mahalanobis) = SqMahalanobis(m.qmat)
_squared_metric(m) = m

_optisim_metric(::Euclidean, cutoff) = (SqEuclidean(), cutoff * cutoff)
_optisim_metric(m::Mahalanobis, cutoff) = (SqMahalanobis(m.qmat), cutoff * cutoff)
_optisim_metric(metric, cutoff) = (metric, cutoff)

function _prune_similar!(candidates, min_dist, cutoff)
  i = 1
  @inbounds while i <= length(candidates)
    if min_dist[candidates[i]] < cutoff
      candidates[i] = candidates[end]
      pop!(candidates)
    else
      i += 1
    end
  end
end

"""
    _blocked_cv_partition(data, k, pre_gap, post_gap; time, name) -> CrossValidationSplit

Shared implementation for contiguous-block k-fold CV strategies. Sorts
observations by `time`, distributes the `k` blocks, and for each fold
uses everything outside `[test_lo - pre_gap, test_hi + post_gap]` as
the train cohort.

`name` is used only in error messages to identify the calling strategy.
"""
function _blocked_cv_partition(
  data,
  k::Int,
  pre_gap::Int,
  post_gap::Int;
  time,
  name::String,
)
  N = numobs(data)
  sorted_dates, order = groupsortperm(time)
  B = length(sorted_dates)

  k <= B || throw(
    SplitParameterError("$name(k=$k) requires at least k distinct time values; got $B."),
  )

  chunk_block_end = distribute_blocks(B, k)
  block_offset = group_offsets(sorted_dates, order, time)

  result = Vector{TrainTestSplit{Vector{Int}}}(undef, k)
  for i = 1:k
    test_block_start = i == 1 ? 1 : chunk_block_end[i-1] + 1
    test_block_end = chunk_block_end[i]

    test_lo = block_offset[test_block_start] + 1
    test_hi = block_offset[test_block_end+1]

    train_left_hi = test_lo - 1 - pre_gap
    train_right_lo = test_hi + 1 + post_gap

    train_left = train_left_hi >= 1 ? order[1:train_left_hi] : Int[]
    train_right = train_right_lo <= N ? order[train_right_lo:N] : Int[]
    train_idx = vcat(train_left, train_right)

    !isempty(train_idx) || throw(
      SplitParameterError(
        "$name: fold $i has empty train cohort " *
        "(pre_gap=$pre_gap, post_gap=$post_gap too large for the surrounding blocks).",
      ),
    )

    result[i] = TrainTestSplit(train_idx, order[test_lo:test_hi])
  end
  return CrossValidationSplit(result)
end
