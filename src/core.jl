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
Base.float(vf::ValidFraction) = vf.frac
Base.convert(::Type{T}, vf::ValidFraction) where {T<:Number} = convert(T, vf.frac)
