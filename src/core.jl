using Random
using ArrayInterface

abstract type SplitStrategy end

"""
    SplitResult

Abstract supertype for all split result types.
"""
abstract type SplitResult end

"""
    TrainTestSplit

A struct representing a train/test split.

# Fields
- `train`: indices of training samples
- `test`: indices of test samples
"""
struct TrainTestSplit{I} <: SplitResult
  train::I
  test::I
end

"""
    TrainValTestSplit

A struct representing a train/validation/test split.

# Fields
- `train`: indices of training samples
- `val`: indices of validation samples
- `test`: indices of test samples
"""
struct TrainValTestSplit{I} <: SplitResult
  train::I
  val::I
  test::I
end

"""
    CrossValidationSplit

A struct representing a k-fold cross-validation split.

# Fields
- `folds`: a vector of TrainTestSplit or TrainValTestSplit
"""
struct CrossValidationSplit{T<:SplitResult} <: SplitResult
  folds::Vector{T}
end


"""
    splitdata(result::SplitResult, X)

Given a SplitResult and data X, return the corresponding splits as a tuple.
"""
function splitdata(result::SplitResult, X)
  throw(
    ErrorException(
      "splitdata is not implemented for $(typeof(result)). Please implement splitdata(::$(typeof(result)), X).",
    ),
  )
end

function splitdata(result::TrainTestSplit, X)
  (X[result.train, :], X[result.test, :])
end

function splitdata(result::TrainValTestSplit, X)
  (X[result.train, :], X[result.val, :], X[result.test, :])
end

function splitdata(result::CrossValidationSplit, X)
  [splitdata(fold, X) for fold in result.folds]
end


"""
    split(data, strategy; rng=Random.default_rng()) -> SplitResult

Split `data` into train/test sets according to `strategy`.
"""
function split(X, strategy::SplitStrategy; rng = Random.default_rng())
  !is_sample_array(typeof(X)) && "Your data type must support sample_indices and get_sample"
  isempty(X) && throw(ArgumentError("Data must not be empty"))
  length(axes(X, 1)) == 1 && throw(ArgumentError("Can not split a single data point"))

  result = _split(X, strategy; rng)
  return result
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

  result = _split((X, y), strategy; rng)
  return result
end

function split(X, y, strategy::SplitStrategy; rng = Random.default_rng())
  !is_sample_array(typeof(X)) && "Your data type must support sample_indices and get_sample"
  isempty(X) && throw(ArgumentError("Data must not be empty"))
  length(axes(X, 1)) == 1 && throw(ArgumentError("Can not split a single data point"))
  size(X, 1) == length(y) ||
    throw(ArgumentError("X and y must have the same number of samples."))

  result = _split(X, y, strategy; rng)
  return result
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

Return a vector of valid sample indices for `A``, supporting all AbstractArray types.

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
  collect(ArrayInterface.axes(A, 1))
end
