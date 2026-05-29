using Distances, LinearAlgebra, Statistics

"""
    XYOnionSplit <: AbstractSplitStrategy

XY-Onion algorithm for train/test splitting.

Combines the SPXY joint X–y distance with the Onion layered-sampling strategy.
The dataset is peeled in `n_layers` concentric shells: in each layer, the outermost
samples (farthest from the centroid in the combined X–y space) are selected for the
training set and the next-outermost for the test set.  Remaining samples after all
layers are assigned randomly in the requested proportion.

The split sizes are approximate: rounding in each layer means the final train/test
counts may differ by a few samples from the requested percentages.

# Fields
- `n_layers::Int`: Number of onion layers (default: `3`)
- `metric_X::Union{Nothing,Distances.SemiMetric}`: Distance metric for `X`.
  `nothing` computes Mahalanobis distance via eigendecomposition of the data
  covariance at split time (equivalent to the `mahalanobis=1` flag in the original
  MATLAB implementation).  `Euclidean()` (default) uses standard Euclidean distance.

# Examples
```julia
res = partition(X, XYOnionSplit(); target = y, train = 70, test = 30)
res = partition(X, XYOnionSplit(; metric_X = nothing); target = y, train = 70, test = 30)
X_train, X_test = splitdata(res, X)
```

# References
Ezenarro, J. et al. (2025). *Analytica Chimica Acta*, 344229.
https://doi.org/10.1016/j.aca.2025.344229

# See also
[`OnionSplit`](@ref) — X-only variant of this algorithm.
"""
struct XYOnionSplit <: AbstractSplitStrategy
  n_layers::Int
  metric_X::Union{Nothing,Distances.SemiMetric}
end

XYOnionSplit(; n_layers::Int = 3, metric_X = Euclidean()) = XYOnionSplit(n_layers, metric_X)

consumes(::XYOnionSplit) = (:data, :target)
fallback_from_data(::XYOnionSplit) = ()

# ---------------------------------------------------------------------------
# Shared infrastructure used by both OnionSplit and XYOnionSplit
# ---------------------------------------------------------------------------

# Whiten X (F×N) so Euclidean distances equal Mahalanobis distances in the original space.
function _xyonion_whiten(X::AbstractMatrix)
  C = cov(X; dims = 2)
  F = eigen(Symmetric(C))
  W = F.vectors * Diagonal(1 ./ sqrt.(max.(F.values, eps()))) * F.vectors'
  return W * X
end

# Allocation-free argmax of ‖X[:,j]‖²  (j ∈ 1:last).
function _onion_stds_argmax(X::AbstractMatrix, last::Int)
  F = size(X, 1)
  best = -Inf
  k = 1
  @inbounds for j = 1:last
    nX = 0.0
    @simd for f = 1:F
      nX = muladd(X[f, j], X[f, j], nX)
    end
    if nX > best
      best = nX
      k = j
    end
  end
  return k
end

# Allocation-free argmax of ‖X[:,j]‖²/max_nX + y[j]²/max_nY  (j ∈ 1:last).
# norms_sq is a pre-allocated scratch buffer of length ≥ last.
function _onion_stds_argmax_xy(
  X::AbstractMatrix,
  y::AbstractVector,
  last::Int,
  norms_sq::AbstractVector,
)
  F = size(X, 1)
  max_nX = 0.0
  max_nY = 0.0
  @inbounds for j = 1:last
    nX = 0.0
    @simd for f = 1:F
      nX = muladd(X[f, j], X[f, j], nX)
    end
    norms_sq[j] = nX
    max_nX = max(max_nX, nX)
    max_nY = max(max_nY, y[j]^2)
  end
  inv_nX = max_nX > 0 ? 1.0 / max_nX : 0.0
  inv_nY = max_nY > 0 ? 1.0 / max_nY : 0.0
  best = -Inf
  k = 1
  @inbounds for j = 1:last
    s = muladd(y[j]^2, inv_nY, norms_sq[j] * inv_nX)
    if s > best
      best = s
      k = j
    end
  end
  return k
end

"""
    _onion_stdsslct(X, y, nosamps) -> Vector{Int}

Select `nosamps` outer samples from mean-centred `X` (F×M) using the
norm-based iterative orthogonalisation variant of the distslct algorithm.

`y` is either an `AbstractVector` (XY-Onion: both X and y distances contribute
to sample scoring) or `nothing` (plain Onion: X distance only).

Each iteration: pick the sample with the largest score, swap it to the last active
position (in-place), orthogonalise the remaining columns against it via BLAS rank-1
update, then shrink the active window.
"""
function _onion_stdsslct(X::AbstractMatrix, y::AbstractVector, nosamps::Int)
  F, M = size(X)
  Xw = float.(X)
  yw = float.(y)
  idx = collect(1:M)
  subset = Vector{Int}(undef, nosamps)
  proj = Vector{Float64}(undef, M)
  norms_sq = Vector{Float64}(undef, M)
  tmp_col = Vector{Float64}(undef, F)

  for i = 1:nosamps
    last = M - i + 1
    k = _onion_stds_argmax_xy(Xw, yw, last, norms_sq)
    subset[i] = idx[k]

    if k != last
      copyto!(tmp_col, view(Xw, :, k))
      copyto!(view(Xw, :, k), view(Xw, :, last))
      copyto!(view(Xw, :, last), tmp_col)
      yw[k], yw[last] = yw[last], yw[k]
      idx[k], idx[last] = idx[last], idx[k]
    end

    if last > 1
      rx0 = view(Xw, :, last)
      nr2 = dot(rx0, rx0)
      if nr2 > 0
        Xrem = view(Xw, :, 1:(last-1))
        pv = view(proj, 1:(last-1))
        BLAS.gemv!('T', 1.0 / nr2, Xrem, rx0, 0.0, pv)
        BLAS.ger!(-1.0, rx0, pv, Xrem)
      end
      yw[1:(last-1)] .-= yw[last]
    end
  end
  return subset
end

function _onion_stdsslct(X::AbstractMatrix, ::Nothing, nosamps::Int)
  F, M = size(X)
  Xw = float.(X)
  idx = collect(1:M)
  subset = Vector{Int}(undef, nosamps)
  proj = Vector{Float64}(undef, M)
  tmp_col = Vector{Float64}(undef, F)

  for i = 1:nosamps
    last = M - i + 1
    k = _onion_stds_argmax(Xw, last)
    subset[i] = idx[k]

    if k != last
      copyto!(tmp_col, view(Xw, :, k))
      copyto!(view(Xw, :, k), view(Xw, :, last))
      copyto!(view(Xw, :, last), tmp_col)
      idx[k], idx[last] = idx[last], idx[k]
    end

    if last > 1
      rx0 = view(Xw, :, last)
      nr2 = dot(rx0, rx0)
      if nr2 > 0
        Xrem = view(Xw, :, 1:(last-1))
        pv = view(proj, 1:(last-1))
        BLAS.gemv!('T', 1.0 / nr2, Xrem, rx0, 0.0, pv)
        BLAS.ger!(-1.0, rx0, pv, Xrem)
      end
    end
  end
  return subset
end

# Add Euclidean distances from sample `k` to `dX` (and optionally `dY`) in-place.
function _onion_add_dists!(
  dX::AbstractVector,
  dY::AbstractVector,
  Xc::AbstractMatrix,
  yc::AbstractVector,
  col_norms_sq::AbstractVector,
  dots::AbstractVector,
  k::Int,
)
  xsel = view(Xc, :, k)
  nsel_sq = dot(xsel, xsel)
  BLAS.gemv!('T', 1.0, Xc, xsel, 0.0, dots)
  @inbounds for j in eachindex(dX)
    dX[j] += sqrt(max(0.0, col_norms_sq[j] - 2 * dots[j] + nsel_sq))
  end
  ysel = yc[k]
  @inbounds for j in eachindex(dY)
    dY[j] += abs(yc[j] - ysel)
  end
end

function _onion_add_dists!(
  dX::AbstractVector,
  ::Nothing,
  Xc::AbstractMatrix,
  ::Nothing,
  col_norms_sq::AbstractVector,
  dots::AbstractVector,
  k::Int,
)
  xsel = view(Xc, :, k)
  nsel_sq = dot(xsel, xsel)
  BLAS.gemv!('T', 1.0, Xc, xsel, 0.0, dots)
  @inbounds for j in eachindex(dX)
    dX[j] += sqrt(max(0.0, col_norms_sq[j] - 2 * dots[j] + nsel_sq))
  end
end

# Allocation-free argmax of dX[j]/max_dX + dY[j]/max_dY.
function _onion_dist_argmax(dX::AbstractVector, dY::AbstractVector)
  max_dX = maximum(dX)
  max_dY = maximum(dY)
  inv_dX = max_dX > 0 ? 1.0 / max_dX : 0.0
  inv_dY = max_dY > 0 ? 1.0 / max_dY : 0.0
  best = -Inf
  k = 1
  @inbounds for j in eachindex(dX)
    s = muladd(dY[j], inv_dY, dX[j] * inv_dX)
    if s > best
      best = s
      k = j
    end
  end
  return k
end

_onion_dist_argmax(dX::AbstractVector, ::Nothing) = argmax(dX)

"""
    _onion_distslct(X, y, nosamps) -> Vector{Int}

Select `nosamps` outer samples from `X` (F×M).

`y` is either an `AbstractVector` (XY-Onion) or `nothing` (plain Onion).
Mean-centres the inputs, then dispatches to `_onion_stdsslct` when `nosamps ≤ F`,
otherwise bootstraps F samples via `_onion_stdsslct` and expands greedily using
cumulative distance, zeroing selected positions before each argmax to prevent
reselection.
"""
function _onion_distslct(X::AbstractMatrix, y, nosamps::Int)
  nosamps == 0 && return Int[]
  F, M = size(X)
  nosamps = min(nosamps, M)

  Xc = X .- mean(X; dims = 2)
  yc = y === nothing ? nothing : y .- mean(y)

  nosamps <= F && return _onion_stdsslct(Xc, yc, nosamps)

  isel = Vector{Int}(undef, nosamps)
  isel[1:F] = _onion_stdsslct(Xc, yc, F)

  selected = falses(M)
  selected[isel[1:F]] .= true

  col_norms_sq = vec(sum(abs2, Xc; dims = 1))
  dX = zeros(M)
  dY = yc === nothing ? nothing : zeros(M)
  dots = Vector{Float64}(undef, M)

  for ii = 1:F
    _onion_add_dists!(dX, dY, Xc, yc, col_norms_sq, dots, isel[ii])
  end
  dX[selected] .= 0.0
  dY !== nothing && (dY[selected] .= 0.0)

  for ii = (F+1):nosamps
    isel[ii] = _onion_dist_argmax(dX, dY)
    selected[isel[ii]] = true
    dX[isel[ii]] = 0.0
    dY !== nothing && (dY[isel[ii]] = 0.0)
    if ii < nosamps
      _onion_add_dists!(dX, dY, Xc, yc, col_norms_sq, dots, isel[ii])
      dX[selected] .= 0.0
      dY !== nothing && (dY[selected] .= 0.0)
    end
  end

  return isel
end

# ---------------------------------------------------------------------------
# Shared _partition loop body
# ---------------------------------------------------------------------------

function _onion_partition!(
  Xw::AbstractMatrix,
  yf,   # AbstractVector or nothing
  n_train::Int,
  n_test::Int,
  n_layers::Int,
  rng,
)
  N = size(Xw, 2)
  fraction = n_train / (n_train + n_test)
  loopfrac = 0.1

  i0 = collect(1:N)
  train_idx = sizehint!(Int[], n_train)
  test_idx = sizehint!(Int[], n_test)

  for _ = 1:n_layers
    m0 = length(i0)
    m0 == 0 && break

    ncal = round(Int, loopfrac * m0 * fraction)
    if ncal > 0
      yi = yf === nothing ? nothing : yf[i0]
      isel = _onion_distslct(Xw[:, i0], yi, min(m0, ncal))
      append!(train_idx, i0[isel])
      deleteat!(i0, sort(isel))
    end

    m0 = length(i0)
    m0 == 0 && break

    ntest = round(Int, loopfrac * m0 * (1 - fraction))
    if ntest > 0
      yi = yf === nothing ? nothing : yf[i0]
      isel = _onion_distslct(Xw[:, i0], yi, min(m0, ntest))
      append!(test_idx, i0[isel])
      deleteat!(i0, sort(isel))
    end
  end

  m0 = length(i0)
  if m0 > 0
    nc = ceil(Int, m0 * fraction)
    labels = shuffle(rng, [trues(nc); falses(m0 - nc)])
    for (j, to_train) in enumerate(labels)
      push!(to_train ? train_idx : test_idx, i0[j])
    end
  end

  return train_idx, test_idx
end

# ---------------------------------------------------------------------------
# XYOnionSplit _partition
# ---------------------------------------------------------------------------

function _partition(
  X,
  s::XYOnionSplit;
  target,
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  Xf = float.(X)
  Xw = s.metric_X === nothing ? _xyonion_whiten(Xf) : Xf
  yf = float.(target)
  train_idx, test_idx = _onion_partition!(Xw, yf, n_train, n_test, s.n_layers, rng)
  return TrainTestSplit(train_idx, test_idx)
end

function _partition(
  X::AbstractVector{<:AbstractVector},
  s::XYOnionSplit;
  target,
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  _partition(stack(X), s; target, n_train, n_test, rng)
end
