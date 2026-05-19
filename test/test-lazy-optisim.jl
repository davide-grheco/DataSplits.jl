using Test
using DataSplits
using Random
using Distances

function _pct(frac)
  train = round(Int, frac * 100)
  test = 100 - train
  return train, test
end

function make_split(
  X;
  frac = 0.7,
  max_subsample_size = 3,
  metric = Euclidean(),
  distance_cutoff = 0.5,
  rng = Random.GLOBAL_RNG,
)
  train, test = _pct(frac)
  return DataSplits.partition(
    X,
    LazyOptiSimSplit(;
      max_subsample_size = max_subsample_size,
      distance_cutoff = distance_cutoff,
      metric = metric,
    );
    train = train,
    test = test,
    rng = rng,
  )
end

@testset "OptiSim splitter (index check)" begin

  X = [
    0 0 0 0 0
    1 0 0 0 0
    1 1 0 0 0
    1 1 1 0 0
    1 1 1 1 0
  ]'
  y = [1, 2, 3, 4, 5]

  result = DataSplits.partition(
    X,
    LazyOptiSimSplit(;
      max_subsample_size = 3,
      distance_cutoff = 0.35,
      metric = Euclidean(),
    );
    train = 80,
    test = 20,
    rng = Random.seed!(42),
  )
  train_idx, test_idx = result.train, result.test

  @test Set(train_idx) == Set([5, 4, 2, 3])
  @test Set(test_idx) == Set([1])

  @test is_disjoint(result)
  @test length(train_idx) + length(test_idx) == 5

  X = randn(10, 50)
  rng1 = MersenneTwister(123)
  result = DataSplits.partition(
    X,
    LazyOptiSimSplit(;
      max_subsample_size = 3,
      distance_cutoff = 0.35,
      metric = Euclidean(),
    );
    train = 60,
    test = 40,
    rng = rng1,
  )
  t1a, te1a = result.train, result.test

  rng2 = MersenneTwister(123)
  result = DataSplits.partition(
    X,
    OptiSimSplit(; max_subsample_size = 3, distance_cutoff = 0.35, metric = Euclidean());
    train = 60,
    test = 40,
    rng = rng2,
  )
  @test t1a == result.train
  @test te1a == result.test

  rng2 = MersenneTwister(123)
  result1b = make_split(X; frac = 0.6, distance_cutoff = 0.35, rng = rng2)
  t1b, te1b = result1b.train, result1b.test
  @test t1a == t1b && te1a == te1b

  rng3 = MersenneTwister(124)
  result2 = make_split(X; frac = 0.6, rng = rng3)
  t2, te2 = result2.train, result2.test
  @test t2 != t1a || te2 != te1a

end
