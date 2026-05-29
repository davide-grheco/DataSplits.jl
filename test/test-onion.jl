using Test
using DataSplits
using Distances
using Random
import DataSplits: SplitInputError

rng_data = MersenneTwister(42)
X50 = randn(rng_data, 4, 50)

@testset "OnionSplit basic properties" begin
  res = partition(X50, OnionSplit(); train = 70, test = 30, rng = MersenneTwister(1))
  train_idx, test_idx = res.train, res.test

  @test length(train_idx) + length(test_idx) == 50
  @test isempty(intersect(Set(train_idx), Set(test_idx)))
  @test Set(vcat(train_idx, test_idx)) == Set(1:50)
  @test !isempty(train_idx)
  @test !isempty(test_idx)
end

@testset "OnionSplit Mahalanobis" begin
  res = partition(
    X50,
    OnionSplit(; metric_X = nothing);
    train = 70,
    test = 30,
    rng = MersenneTwister(1),
  )
  all_idx = vcat(res.train, res.test)

  @test length(all_idx) == 50
  @test Set(all_idx) == Set(1:50)
end

@testset "OnionSplit n_layers parameter" begin
  for nl in (1, 5)
    res = partition(
      X50,
      OnionSplit(; n_layers = nl);
      train = 70,
      test = 30,
      rng = MersenneTwister(1),
    )
    @test length(res.train) + length(res.test) == 50
    @test Set(vcat(res.train, res.test)) == Set(1:50)
  end
end

@testset "OnionSplit does not require target" begin
  @test_nowarn partition(X50, OnionSplit(); train = 70, test = 30)
end

@testset "OnionSplit target keyword rejected" begin
  y = randn(50)
  @test_throws SplitInputError partition(
    X50,
    OnionSplit();
    target = y,
    train = 70,
    test = 30,
  )
end

@testset "OnionSplit vector-of-vectors input" begin
  Xvov = [X50[:, i] for i = 1:50]
  res = partition(Xvov, OnionSplit(); train = 70, test = 30, rng = MersenneTwister(1))
  @test length(res.train) + length(res.test) == 50
  @test Set(vcat(res.train, res.test)) == Set(1:50)
end

@testset "OnionSplit nosamps > F (SPXY extension path)" begin
  X3 = randn(MersenneTwister(7), 3, 50)
  res = partition(
    X3,
    OnionSplit(; n_layers = 5);
    train = 70,
    test = 30,
    rng = MersenneTwister(1),
  )
  @test length(res.train) + length(res.test) == 50
  @test Set(vcat(res.train, res.test)) == Set(1:50)
end
