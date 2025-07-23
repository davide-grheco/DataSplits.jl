#############################################################
#  SPXY (Sample set Partitioning based on joint X–Y distance)
#  – in‑memory version (O(N²))
#############################################################

using Distances

struct SPXYSplit{T,M<:Distances.SemiMetric} <: SplitStrategy
  frac::ValidFraction{T}
  metric::M
end

"""
    SPXYSplit(frac; metric = Euclidean())

Create an **SPXY splitter** – the variant of Kennard–Stone in which
the distance matrix is the *element‑wise sum* of

* the (normalised) pairwise distance matrix of the feature matrix `X`
* **plus** the (normalised) pairwise distance matrix of the response
  vector `y`.

`frac` is the fraction of samples that will end up in the **training**
subset.

!!! note
    `split` **must** be called with a 2‑tuple `(X, y)` or with
    positional arguments `split(X, y, strategy)`;
    calling `split(X, strategy)` will raise a `MethodError`, because `y`
    is mandatory for SPXY.

### Arguments
| name   | type                       | meaning                                    |
|:------ |:-------------------------- |:------------------------------------------ |
| `frac` | `Real` (0 < `frac` < 1)    | training‑set fraction                      |
| `metric` | [`Distances.SemiMetric`] | distance metric used for both `X` and `y`  |

### See also
[`KennardStoneSplit`](@ref) — the classical variant that uses only `X`.
"""
SPXYSplit(frac::Real; metric = Euclidean()) = SPXYSplit(ValidFraction(frac), metric)

const MDKSSplit = SPXYSplit
MDKSSplit(frac::Real) = SPXYSplit(frac; metric = Mahalanobis())

@inline function _norm_pairwise(mat::AbstractMatrix, metric)
  D = pairwise(metric, mat, mat; dims = 1)
  return D ./ maximum(D)
end

@inline function _norm_pairwise(y::AbstractVector, metric)
  Y = reshape(y, :, 1)
  D = pairwise(metric, Y, Y; dims = 1)
  return D ./ maximum(D)
end

"""
    _split(X, y, strategy::SPXYSplit; rng = Random.GLOBAL_RNG) → (train_idx, test_idx)

Split a **feature matrix `X`** and **response vector `y`** into
train/test subsets using the SPXY algorithm:

1.  Build the joint distance matrix `D = D_X + D_Y`
    (see [`SPXYSplit`](@ref) for details).
2.  Run the Kennard–Stone maximin procedure on `D`.
3.  Return two **sorted** index vectors (`train_idx`, `test_idx`).

### Arguments
| name      | type                 | requirement |
|:--------- |:-------------------- |:----------- |
| `X`       | `AbstractMatrix`     | `size(X, 1) == length(y)` |
| `y`       | `AbstractVector`     |             |
| `strategy`| `SPXYSplit`          | created with `SPXYSplit(frac; metric)` |
| `rng`     | random‑number source | *unused* here but kept for API symmetry |

### Returns
Two `Vector{Int}` with the **row indices** of `X` (and the corresponding
entries of `y`) that belong to the training and test subsets.

The indices are *axis‑correct* — if `X` is an `OffsetMatrix` whose first
row is index `0`, the returned indices will also start at `0`.
"""
function _split(
  X::AbstractArray,
  y::AbstractVector,
  strategy::SPXYSplit;
  rng = Random.GLOBAL_RNG,
)
  idx_range = axes(X, 1)
  N = length(idx_range)

  n_test = round(Int, (1 - strategy.frac) * N)
  n_train = N - n_test

  DX = _norm_pairwise(X, strategy.metric)
  DY = _norm_pairwise(y, strategy.metric)
  D = DX .+ DY

  train_idx, test_idx = kennard_stone_from_distance_matrix(D, n_train)

  return train_idx, test_idx
end

# convenience method when the caller passes a tuple
_split((X, y), strategy::SPXYSplit; kwargs...) = _split(X, y, strategy; kwargs...)
