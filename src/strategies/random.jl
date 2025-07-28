using Random

export RandomSplit

"""
    RandomSplit{T} <: SplitStrategy

Randomly splits data into train/test sets according to the specified fraction.

# Fields
- `frac::ValidFraction{T}`: Fraction of data to use for training (0 < frac < 1)

# Examples
```julia
splitter = RandomSplit(0.8)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)
```
"""
struct RandomSplit{T} <: SplitStrategy
  frac::ValidFraction{T}
end
RandomSplit(frac::Real) = RandomSplit(ValidFraction(frac))

function _split(data, s::RandomSplit; rng)
  N = numobs(data)
  perm = randperm(rng, N)
  cut = floor(Int, s.frac * N)
  train_pos = perm[1:cut]
  test_pos = perm[cut+1:end]
  return TrainTestSplit(train_pos, test_pos)
end
