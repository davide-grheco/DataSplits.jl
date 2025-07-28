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

A result type representing a train/test split.

# Fields
- `train::Vector{Int}`: Indices (1:N) of training samples.
- `test::Vector{Int}`: Indices (1:N) of test samples.

# Notes
- Indices are always in the range 1:N, where N is the number of samples.
- For custom data types, use `getobs(X, indices)` to access split data.
- Data matrices are expected to have columns as samples (features × samples).

# Examples
```julia
result = split(X, KennardStoneSplit(0.8))
X_train, X_test = splitdata(result, X)
```
"""
struct TrainTestSplit{I} <: SplitResult
  train::I
  test::I
end

"""
    TrainValTestSplit

A result type representing a train/validation/test split.

# Fields
- `train::Vector{Int}`: Indices (1:N) of training samples.
- `val::Vector{Int}`: Indices (1:N) of validation samples.
- `test::Vector{Int}`: Indices (1:N) of test samples.

# Notes
- Indices are always in the range 1:N, where N is the number of samples.
- For custom data types, use `getobs(X, indices)` to access split data.
- Data matrices are expected to have columns as samples (features × samples).

# Examples
```julia
result = split(X, SomeTrainValTestSplit(...))
X_train, X_val, X_test = splitdata(result, X)
```
"""
struct TrainValTestSplit{I} <: SplitResult
  train::I
  val::I
  test::I
end

"""
    CrossValidationSplit

A result type representing a k-fold cross-validation split.

# Fields
- `folds::Vector{<:SplitResult}`: Vector of TrainTestSplit or TrainValTestSplit, one per fold.

# Notes
- Each fold contains indices in the range 1:N, where N is the number of samples.
- For custom data types, use `getobs(X, indices)` to access split data.
- Data matrices are expected to have columns as samples (features × samples).

# Examples
```julia
result = split(X, SomeCVSplit(...))
for (X_train, X_test) in splitdata(result, X)
    # ...
end
```
"""
struct CrossValidationSplit{T<:SplitResult} <: SplitResult
  folds::Vector{T}
end


"""
    splitdata(result::SplitResult, X)

Return the train/test (and optionally validation) splits from a `SplitResult` for the given data.

# Arguments
- `result::SplitResult`: The result of a splitting strategy.
- `X`: Data matrix or custom container. Columns are samples.

# Returns
- Tuple of data splits, e.g. `(X_train, X_test)`.

# Notes
- All indices in `result` are in the range 1:N, where N is the number of samples.
- For custom data types, use `getobs(X, indices)` to access split data.
- Data matrices are expected to have columns as samples (features × samples).

# Examples
```julia
result = split(X, KennardStoneSplit(0.8))
X_train, X_test = splitdata(result, X)
```
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

Split data into train/test (or train/val/test, or cross-validation) sets according to the given strategy.

# Arguments
- `data`: Data matrix (columns are samples) or custom container.
- `strategy::SplitStrategy`: The splitting strategy to use.
- `rng`: Optional random number generator.

# Returns
- `SplitResult`: An object containing the split indices.

# Notes
- All returned indices are in the range 1:N, where N is the number of samples.
- For custom data types, implement `Base.length` and `Base.getindex` as per MLUtils.

# Examples
```julia
result = split(X, KennardStoneSplit(0.8))
X_train, X_test = splitdata(result, X)
```
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
