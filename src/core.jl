using Random

export SplitStrategy, split

abstract type SplitStrategy end

"""
    split(data, strategy; rng=Random.default_rng()) → (train, test)

Split `data` into train/test sets according to `strategy`.
"""
function split(X, strategy::SplitStrategy; rng = Random.default_rng())
  !is_sample_array(typeof(X)) && "Your data type must support sample_indices and get_sample"
  isempty(X) && throw(ArgumentError("Data must not be empty"))
  length(axes(X, 1)) == 1 && throw(ArgumentError("Can not split a single data point"))

  _split(X, strategy; rng)
end

function split(
  data::Tuple{AbstractArray,AbstractVector},
  strategy::SplitStrategy;
  rng = Random.default_rng(),
)
  X, y = data
  !is_sample_array(typeof(X)) && "Your data type must support sample_indices and get_sample"
  isempty(X) && throw(ArgumentError("Data must not be empty"))
  size(X, 1) == 1 && throw(ArgumentError("Cannot split a single data point"))
  size(X, 1) == length(y) ||
    throw(ArgumentError("X and y must have the same number of samples."))

  _split((X, y), strategy; rng)
end

function split(X, y, strategy::SplitStrategy; rng = Random.default_rng())
  !is_sample_array(typeof(X)) && "Your data type must support sample_indices and get_sample"
  isempty(X) && throw(ArgumentError("Data must not be empty"))
  length(axes(X, 1)) == 1 && throw(ArgumentError("Can not split a single data point"))
  size(X, 1) == length(y) ||
    throw(ArgumentError("X and y must have the same number of samples."))

  _split(X, y, strategy; rng)
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
    is_sample_array(::Type{T}) -> Bool

Trait method that tells whether type `T` behaves like a sample‑indexed array
that supports `sample_indices`, `_get_sample`, etc.

You may extend this for custom containers to ensure compatibility
with OptiSim or other sampling methods.

Defaults to `false` unless specialized.
"""
is_sample_array(::Type) = false
is_sample_array(::Type{<:AbstractArray}) = true

"""
    get_sample(A::AbstractArray, idx)

Public API for getting samples that handles any valid index type.
Dispatches to _internal_get_sample after index conversion.
"""
function get_sample(A::AbstractArray, idx)
  A[sample_indices(A)[idx], :]
end

function get_sample(A::AbstractVector, idx)
  A[sample_indices(A)[idx]]
end

"""
    sample_indices(A::AbstractArray) -> AbstractVector

Return the list of sample-level indices used to address elements in `A`.

This method defines the "sample axis" (typically axis 1) and determines how your
splitting/sampling algorithms enumerate data points.

## Default
For standard arrays, returns `axes(A, 1)`.

## Extension
To support non-standard arrays (e.g., views, custom wrappers),
you may extend this method to expose logical sample indices:

```julia
Base.sample_indices(a::MyFancyArray) = 1:length(a.ids)
```
"""
@inline function sample_indices(A::AbstractArray)
  axes(A, 1)
end
