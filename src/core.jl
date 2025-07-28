using Random
using ArrayInterface
using MLUtils: numobs, getobs

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

# Notes
- X is expected to have **columns as samples** (features × samples).
- For custom data types, implement `Base.length` (number of samples) and `Base.getindex(data, i)` (returning the i-th sample) as described in the MLUtils documentation.
"""
function splitdata(result::SplitResult, X)
  throw(
    ErrorException(
      "splitdata is not implemented for $(typeof(result)). Please implement splitdata(::$(typeof(result)), X).",
    ),
  )
end

function splitdata(result::TrainTestSplit, X)
  (getobs(X, result.train), getobs(X, result.test))
end

function splitdata(result::TrainValTestSplit, X)
  (getobs(X, result.train), getobs(X, result.val), getobs(X, result.test))
end

function splitdata(result::CrossValidationSplit, X)
  [splitdata(fold, X) for fold in result.folds]
end


"""
    split(data, strategy; rng=Random.default_rng()) -> SplitResult

Split `data` into train/test sets according to `strategy`.
"""
function split(X, strategy::SplitStrategy; rng = Random.default_rng())
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
  isempty(X) && throw(ArgumentError("Data must not be empty"))
  size(X, 1) == 1 && throw(ArgumentError("Cannot split a single data point"))
  size(X, 1) == length(y) ||
    throw(ArgumentError("X and y must have the same number of samples."))

  result = _split((X, y), strategy; rng)
  return result
end

function split(X, y, strategy::SplitStrategy; rng = Random.default_rng())
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
