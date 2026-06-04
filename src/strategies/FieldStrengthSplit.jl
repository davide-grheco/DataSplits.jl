using Distances

"""
    FieldStrengthSplit <: AbstractSplitStrategy

Field-strength-based sample selection for train/test splitting.

Inspired by the spatial field-strength distribution law from electrostatics
(He et al. 2026): each sample is treated as a point charge generating a field
in its neighbourhood, with the field strength at any position being the sum of
inverse-squared-distance contributions from all already-selected samples.

The algorithm iteratively selects the sample with the **minimum** accumulated
field strength — i.e. the sample least represented by the current selection —
building a calibration set that spreads progressively from the most isolated
regions inward.

Compared to [`KennardStoneSplit`](@ref) (which maximises the *minimum* distance
to the nearest already-selected sample), the field-strength criterion accounts
for the *cumulative influence* of all selected samples, producing a smoother
distribution that is less dominated by a single nearest neighbour.

# Fields
- `metric::Distances.SemiMetric`: Distance metric (default: `Euclidean()`)

# Notes
- Split sizes are exact.
- Precomputes the full N×N distance matrix (O(N²) memory).

# Examples
```julia
res = partition(X, FieldStrengthSplit(); train = 0.7, test = 0.3)
X_train, X_test = splitdata(res, X)
```

# References
He, Z. et al. A new sample selection method based on field strength distribution
for developing near-infrared spectroscopy calibration models.
*Journal of Chemometrics* 2026, e70094.
<https://doi.org/10.1002/cem.70094>.
"""
struct FieldStrengthSplit <: AbstractSplitStrategy
  metric::Distances.SemiMetric
end

FieldStrengthSplit() = FieldStrengthSplit(Euclidean())

consumes(::FieldStrengthSplit) = (:data,)
fallback_from_data(::FieldStrengthSplit) = ()

const _FS_EPS = 1e-10  # guard against zero distances (duplicate samples)

"""
    field_strength_from_distance_matrix(D, n_train) -> (train, test)

Field-strength selection on a precomputed N×N distance matrix `D`.
Returns index vectors of length `n_train` and `N - n_train`.
"""
function field_strength_from_distance_matrix(D::AbstractMatrix, n_train::Integer)
  N = size(D, 1)

  # Isolation score for each sample: sum of 1/d² to all others.
  # Low isolation = most isolated from the full dataset → best first sample.
  iso = zeros(N)
  @inbounds for j = 1:(N-1)
    col = @view D[(j+1):N, j]
    for i = 1:(N-j)
      c = 1 / max(col[i], _FS_EPS)^2
      iso[j+i] += c
      iso[j] += c
    end
  end

  selected = Vector{Int}(undef, n_train)
  selected[1] = argmin(iso)

  col1 = @view D[:, selected[1]]
  field = Vector{Float64}(undef, N)
  @inbounds @simd for m = 1:N
    field[m] = 1 / max(col1[m], _FS_EPS)^2
  end
  field[selected[1]] = Inf

  for t = 2:n_train
    k = argmin(field)
    selected[t] = k
    col_k = @view D[:, k]
    @inbounds @simd for m = 1:N
      field[m] += 1 / max(col_k[m], _FS_EPS)^2
    end
    field[k] = Inf
  end

  return selected, setdiff(1:N, selected)
end

function _partition(
  X,
  s::FieldStrengthSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  D = distance_matrix(X, s.metric)
  train, test = field_strength_from_distance_matrix(D, n_train)
  return TrainTestSplit(train, test)
end

function _partition(
  X::AbstractVector{<:AbstractVector},
  s::FieldStrengthSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  _partition(stack(X), s; n_train, n_test, rng)
end
