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
  strategy = RandomSplit(0.6)
  result = partition(data_std, strategy; rng)
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
    result = partition(data_offset, strategy; rng)
    train_idx, test_idx = result.train, result.test

    @test length(train_idx) == 6
    @test length(test_idx) == 4
    @test all(1 .≤ train_idx .≤ 10)
    @test all(1 .≤ test_idx .≤ 10)
    @test isempty(intersect(train_idx, test_idx))
  end

  @testset "Edge Cases" begin
    @test_throws SplitInputError partition(zeros(2, 0), strategy; rng)
    @test_throws SplitParameterError partition(rand(2, 10), RandomSplit(0.0); rng)
    @test_throws SplitParameterError partition(rand(2, 10), RandomSplit(1.0); rng)
  end

  @testset "Randomness" begin
    rng = MersenneTwister(123)
    result = partition(data_std, strategy; rng)
    train1, test1 = result.train, result.test
    rng = MersenneTwister(123)
    result = partition(data_std, strategy; rng)
    train2, test2 = result.train, result.test
    @test train1 == train2 && test1 == test2

    rng2 = MersenneTwister(223)
    result = partition(data_std, strategy; rng = rng2)
    train3, test3 = result.train, result.test
    @test train1 != train3
  end
end
