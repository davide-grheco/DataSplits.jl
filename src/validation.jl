
"""
    ValidFraction{T<:Real}

A wrapper type guaranteeing a real value strictly in (0, 1).

Arithmetic with plain numbers delegates to the underlying value so it can be
used transparently in formulas.
"""
struct ValidFraction{T<:Real}
  frac::T
  function ValidFraction(frac::T) where {T<:Real}
    _is_fraction(frac) ||
      throw(SplitParameterError("Fraction must be strictly between 0 and 1. Got $frac."))
    new{T}(frac)
  end
end

Base.:*(vf::ValidFraction, x::Number) = vf.frac * x
Base.:*(x::Number, vf::ValidFraction) = x * vf.frac
Base.:+(x::Number, vf::ValidFraction) = x + vf.frac
Base.:+(x::ValidFraction, y::ValidFraction) = x.frac + y.frac
Base.:+(x::ValidFraction, y::Number) = x.frac + y
Base.:-(x::Number, vf::ValidFraction) = x - vf.frac
Base.float(vf::ValidFraction) = vf.frac
Base.convert(::Type{T}, vf::ValidFraction) where {T<:Number} = convert(T, vf.frac)


_as_valid_fraction(x::ValidFraction) = x
_as_valid_fraction(x::Real) = ValidFraction(x)
_as_valid_fraction(::Nothing) = nothing

"""
Check whether a number is strictly in (0, 1).
"""
_is_fraction(x::Real) = 0 < x < 1

"""
    _assert_unit_fraction_sum(fractions::ValidFraction...)

Assert that the supplied validated fractions form a complete partition.

Throws `SplitParameterError` if the sum of the wrapped fraction values is not
approximately equal to `1`.
"""
function _assert_unit_fraction_sum(fractions::Union{ValidFraction,Nothing}...)
  s = sum(float, filter(!isnothing, fractions))

  isapprox(s, one(s)) ||
    throw(SplitParameterError("Fractional cohort sizes must sum to 1.0; got $s."))

  return nothing
end

function _assert_positive_integer(name::Symbol, value::Integer)
  value >= 1 ||
    throw(SplitParameterError("`$name` must be a positive integer, got $value."))

  return nothing
end

function _assert_positive_integer_parts(parts::Pair{Symbol,<:Integer}...)
  for (name, value) in parts
    _assert_positive_integer(name, value)
  end

  return nothing
end

"""
    _assert_partitionable(data) -> N

Validate that `data` is non-empty and has at least 2 observations.
Returns `numobs(data)` for downstream use. Used by every `partition`
method to guarantee a meaningful split is possible.
"""
function _assert_partitionable(data)
  isempty(data) &&
    throw(SplitInputError("Data must not be empty. Please provide a non-empty dataset."))
  N = numobs(data)
  N >= 2 || throw(SplitInputError("Cannot split fewer than 2 observations."))
  return N
end
