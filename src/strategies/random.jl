using Random

export RandomSplit

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
