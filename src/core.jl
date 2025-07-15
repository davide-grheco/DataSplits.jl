using Random

export SplitStrategy, split

abstract type SplitStrategy end

"""
    split(data, strategy; rng=Random.default_rng()) → (train, test)

Split `data` into train/test sets according to `strategy`.
"""
function split(data, strategy::SplitStrategy; rng=Random.default_rng())
  isempty(data) && throw(ArgumentError("Data must not be empty"))
  length(axes(data, 1)) == 1 && throw(ArgumentError("Can not split a single data point"))

  _split(data, strategy; rng)
end

struct ValidFraction{T<:Real}
  frac::T
  function ValidFraction(frac::T) where {T<:Real}
    if !(0 < frac < 1)
      throw(ArgumentError("Fraction must be between 0 and 1, got $frac"))
    end
    new{T}(frac)
  end
end

Base.:*(vf::ValidFraction, x::Number) = vf.frac * x
Base.:*(x::Number, vf::ValidFraction) = x * vf.frac
Base.:+(x::Number, vf::ValidFraction) = x + vf.frac
Base.:-(x::Number, vf::ValidFraction) = x - vf.frac
Base.float(vf::ValidFraction) = vf.frac
Base.convert(::Type{T}, vf::ValidFraction) where {T<:Number} = convert(T, vf.frac)

"""
    _get_sample(A::AbstractArray, i)

Get the i-th sample from an array along the first dimension, returning a view when possible.
For matrices this returns a row, for 3D arrays a matrix slice, etc.
"""
@inline function _get_sample(A::AbstractArray, i)
  idxs = ntuple(d -> d == 1 ? i : (:), ndims(A))
  A[idxs...]
end

"""
    _sort_indices(range, indices) -> Vector{eltype(range)}

Returns sorted indices while preserving the original index type (Int, Offset, etc.).
Memory-efficient for contiguous ranges.
"""
@inline function _sort_indices!(range::AbstractRange, indices)
  sort!(collect(indices))
end

@inline function _sort_indices!(range, indices)
  sort!([range[i] for i in indices])
end
