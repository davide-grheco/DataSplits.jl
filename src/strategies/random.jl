using Random

export RandomSplit

struct RandomSplit{T} <: SplitStrategy
  frac::ValidFraction{T}
end
RandomSplit(frac::Real) = RandomSplit(ValidFraction(frac))

function random(N, s, rng, data)
  perm = randperm(rng, N)
  cut = floor(Int, s.frac * N)
  train_pos = perm[1:cut]
  test_pos = perm[cut+1:end]
  return TrainTestSplit(train_pos, test_pos)
end

function _split(data, s::RandomSplit; rng)
  split_with_positions(data, s, random; rng = rng)
end
