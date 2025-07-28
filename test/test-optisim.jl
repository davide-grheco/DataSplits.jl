using Test
using DataSplits
using Random
using Distances

function sane_split_check(result, N; ntrain_expected = nothing)
  train, test = result.train, result.test
  @test isempty(intersect(train, test))
  @test length(train) + length(test) == N
  ntrain_expected === nothing || @test length(train) == ntrain_expected
end

function make_split(
  X;
  frac = 0.7,
  max_subsample_size = 3,
  metric = Euclidean(),
  distance_cutoff = 0.5,
  rng = Random.GLOBAL_RNG,
)
  return DataSplits.split(
    X,
    OptiSimSplit(
      frac;
      max_subsample_size = max_subsample_size,
      distance_cutoff = distance_cutoff,
      metric = metric,
    );
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

  result = DataSplits.split(
    X,
    OptiSimSplit(
      0.75;
      max_subsample_size = 3,
      distance_cutoff = 0.35,
      metric = Euclidean(),
    ),
    rng = Random.seed!(42),
  )
  train_idx, test_idx = result.train, result.test

  @test Set(train_idx) == Set([1, 3, 4, 5])
  @test Set(test_idx) == Set([2])

  @test isempty(intersect(train_idx, test_idx))
  @test length(train_idx) + length(test_idx) == 5

  X = randn(10, 50)
  rng1 = MersenneTwister(123)
  result = DataSplits.split(
    X,
    OptiSimSplit(0.6; max_subsample_size = 3, distance_cutoff = 0.35, metric = Euclidean()),
    rng = rng1,
  )
  t1a, te1a = result.train, result.test

  rng2 = MersenneTwister(123)
  result1b = make_split(X; frac = 0.6, distance_cutoff = 0.35, rng = rng2)
  t1b, te1b = result1b.train, result1b.test
  @test t1a == t1b && te1a == te1b
  sane_split_check(result, 50; ntrain_expected = 30)

  rng3 = MersenneTwister(124)
  result2 = make_split(X; frac = 0.6, rng = rng3)
  t2, te2 = result2.train, result2.test
  @test t2 != t1a || te2 != te1a
  sane_split_check(result2, 50)

  Xsmall = randn(3, 8)
  result_small =
    make_split(Xsmall; frac = 0.5, distance_cutoff = 0.35, max_subsample_size = 20)
  tr, te = result_small.train, result_small.test
  sane_split_check(result_small, 8; ntrain_expected = 4)

  result = make_split(X; frac = 0.7, distance_cutoff = 0.35, max_subsample_size = 0)
  sane_split_check(result, 50)

  result = make_split(X; frac = 0.7, distance_cutoff = 0.35, max_subsample_size = 999)
  sane_split_check(result, 50)


  Xvv = [randn(10) for _ = 1:60]
  result = make_split(Xvv; frac = 0.25, distance_cutoff = 0.10, metric = CosineDist())
  sane_split_check(result, 60; ntrain_expected = 15)

  result = make_split(X; frac = 0.8, distance_cutoff = 0)
  sane_split_check(result, 50; ntrain_expected = 40)

  result = make_split(X; frac = 0.8, distance_cutoff = 2000)
  sane_split_check(result, 50; ntrain_expected = 1)

end
