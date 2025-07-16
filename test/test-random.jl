using Random
using OffsetArrays
using DataSplits: split, SplitStrategy, RandomSplit


rng = MersenneTwister(42)

@testset "split() with RandomSplit" begin
  data_std = rand(10, 2)
  strategy = RandomSplit(0.6)
  train_idx, test_idx = split(data_std, strategy; rng)

  @testset "Standard Array" begin
    @test length(train_idx) == 6
    @test length(test_idx) == 4
    @test all(1 .≤ train_idx .≤ 10)
    @test all(1 .≤ test_idx .≤ 10)
    @test isempty(intersect(train_idx, test_idx))
  end

  # ---- 2. Offset Arrays (custom indexing) ----
  data_offset = OffsetArray(rand(10, 2), -5:4, 1:2)
  train_idx, test_idx = split(data_offset, strategy; rng)

  @testset "Offset Array" begin
    @test length(train_idx) == 6
    @test length(test_idx) == 4
    @test all(-5 .≤ train_idx .≤ 4)
    @test all(-5 .≤ test_idx .≤ 4)
    @test isempty(intersect(train_idx, test_idx))
  end

  # ---- 3. Edge Cases ----
  @testset "Edge Cases" begin
    # Tiny array (N < 2)
    @test_throws ArgumentError split(rand(1, 2), strategy; rng)

    # Empty array
    @test_throws ArgumentError split(zeros(0, 2), strategy; rng)

    # Fraction bounds (0 < frac < 1)
    @test_throws ArgumentError split(rand(10, 2), RandomSplit(0.0); rng)
    @test_throws ArgumentError split(rand(10, 2), RandomSplit(1.0); rng)
  end

  # ---- 4. Randomness & Correctness ----
  @testset "Randomness" begin
    # Same RNG → same split
    rng = MersenneTwister(123)
    train1, test1 = split(data_std, strategy; rng)
    rng = MersenneTwister(123)
    train2, test2 = split(data_std, strategy; rng)
    @test train1 == train2 && test1 == test2

    # Different RNG → different split
    rng2 = MersenneTwister(223)
    train3, test3 = split(data_std, strategy; rng = rng2)
    @test train1 != train3
  end
end
