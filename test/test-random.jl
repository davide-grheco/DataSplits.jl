using Random
using OffsetArrays
using DataSplits:
  partition,
  AbstractSplitStrategy,
  RandomSplit,
  SplitInputError,
  SplitParameterError,
  SplitNotImplementedError


rng = MersenneTwister(42)

@testset "partition() with RandomSplit" begin
  data_std = rand(2, 10)
  strategy = RandomSplit()
  result = partition(data_std, strategy; train = 60, test = 40, rng)
  train_idx, test_idx = result.train, result.test

  @testset "Standard Array" begin
    @test length(train_idx) == 6
    @test length(test_idx) == 4
    @test all(1 .≤ train_idx .≤ 10)
    @test all(1 .≤ test_idx .≤ 10)
    @test isempty(intersect(train_idx, test_idx))
  end

  @testset "Offset Array" begin
    data_offset = OffsetArray(rand(2, 10), 1:2, -5:4)
    result = partition(data_offset, strategy; train = 60, test = 40, rng)
    train_idx, test_idx = result.train, result.test

    @test length(train_idx) == 6
    @test length(test_idx) == 4
    @test all(1 .≤ train_idx .≤ 10)
    @test all(1 .≤ test_idx .≤ 10)
    @test isempty(intersect(train_idx, test_idx))
  end

  @testset "Edge Cases" begin
    @test_throws SplitInputError partition(
      zeros(2, 0),
      strategy;
      train = 60,
      test = 40,
      rng,
    )
    @test_throws SplitParameterError partition(
      rand(2, 10),
      strategy;
      train = 0,
      test = 100,
      rng,
    )
    @test_throws SplitParameterError partition(
      rand(2, 10),
      strategy;
      train = 100,
      test = 0,
      rng,
    )
    # Sum neither 100 nor N
    @test_throws SplitParameterError partition(
      rand(2, 10),
      strategy;
      train = 50,
      test = 40,
      rng,
    )
  end

  @testset "Randomness" begin
    rng = MersenneTwister(123)
    result = partition(data_std, strategy; train = 60, test = 40, rng)
    train1, test1 = result.train, result.test
    rng = MersenneTwister(123)
    result = partition(data_std, strategy; train = 60, test = 40, rng)
    train2, test2 = result.train, result.test
    @test train1 == train2 && test1 == test2

    rng2 = MersenneTwister(223)
    result = partition(data_std, strategy; train = 60, test = 40, rng = rng2)
    train3, test3 = result.train, result.test
    @test train1 != train3
  end

  @testset "Absolute counts" begin
    rng = MersenneTwister(7)
    result = partition(data_std, strategy; train = 7, test = 3, rng)
    @test length(result.train) == 7
    @test length(result.test) == 3
  end
end
