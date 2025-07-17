using Test
using DataSplits
using Random
using Distances

function sane_split_check(train, test, N; ntrain_expected = nothing)
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
  ]
  y = [1, 2, 3, 4, 5]

  train_idx, test_idx = DataSplits.split(
    X,
    OptiSimSplit(
      0.75;
      max_subsample_size = 3,
      distance_cutoff = 0.35,
      metric = Euclidean(),
    ),
    rng = Random.seed!(42),
  )

  @test Set(train_idx) == Set([1, 3, 4, 5])
  @test Set(test_idx) == Set([2])

  @test isempty(intersect(train_idx, test_idx))
  @test length(train_idx) + length(test_idx) == size(X, 1)

  X = randn(50, 10)
  rng1 = MersenneTwister(123)
  t1a, te1a = DataSplits.split(
    X,
    OptiSimSplit(0.6; max_subsample_size = 3, distance_cutoff = 0.35, metric = Euclidean()),
    rng = rng1,
  )
  rng2 = MersenneTwister(123)
  t1b, te1b = make_split(X; frac = 0.6, distance_cutoff = 0.35, rng = rng2)
  @test t1a == t1b && te1a == te1b
  sane_split_check(t1a, te1a, 50; ntrain_expected = 30)

  rng3 = MersenneTwister(124)
  t2, te2 = make_split(X; frac = 0.6, rng = rng3)
  @test t2 != t1a || te2 != te1a
  sane_split_check(t2, te2, 50)

  Xsmall = randn(8, 3)
  tr, te = make_split(Xsmall; frac = 0.5, distance_cutoff = 0.35, max_subsample_size = 20)
  sane_split_check(tr, te, 8; ntrain_expected = 4)

  trA, teA = make_split(X; frac = 0.7, distance_cutoff = 0.35, max_subsample_size = 0)
  trB, teB = make_split(X; frac = 0.7, distance_cutoff = 0.35, max_subsample_size = 999)
  sane_split_check(trA, teA, 50)
  sane_split_check(trB, teB, 50)

  Xvv = [randn(10) for _ = 1:60]
  trvv, tevv = make_split(Xvv; frac = 0.25, distance_cutoff = 0.10, metric = CosineDist())
  sane_split_check(trvv, tevv, 60; ntrain_expected = 15)

  trtight, tetight = make_split(X; frac = 0.8, distance_cutoff = 0)
  sane_split_check(trtight, tetight, 50; ntrain_expected = 40)

  trtight, tetight = make_split(X; frac = 0.8, distance_cutoff = 2000)
  sane_split_check(trtight, tetight, 50; ntrain_expected = 1)

end
