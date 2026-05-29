using Test
using DataSplits
using Distances
using Statistics
using Random
import DataSplits: SplitInputError

# Reproducible dataset: 50 samples, 4 features, 1 target
rng_data = MersenneTwister(42)
X50 = randn(rng_data, 4, 50)
y50 = randn(rng_data, 50)

@testset "XYOnion basic properties" begin
  res = partition(
    X50,
    XYOnionSplit();
    target = y50,
    train = 70,
    test = 30,
    rng = MersenneTwister(1),
  )
  train_idx, test_idx = res.train, res.test

  @test length(train_idx) + length(test_idx) == 50
  @test isempty(intersect(Set(train_idx), Set(test_idx)))
  @test Set(vcat(train_idx, test_idx)) == Set(1:50)
  @test !isempty(train_idx)
  @test !isempty(test_idx)
end

@testset "XYOnion Mahalanobis" begin
  res = partition(
    X50,
    XYOnionSplit(; metric_X = nothing);
    target = y50,
    train = 70,
    test = 30,
    rng = MersenneTwister(1),
  )
  train_idx, test_idx = res.train, res.test

  @test length(train_idx) + length(test_idx) == 50
  @test isempty(intersect(Set(train_idx), Set(test_idx)))
  @test Set(vcat(train_idx, test_idx)) == Set(1:50)
end

@testset "XYOnion n_layers parameter" begin
  res1 = partition(
    X50,
    XYOnionSplit(; n_layers = 1);
    target = y50,
    train = 70,
    test = 30,
    rng = MersenneTwister(1),
  )
  res5 = partition(
    X50,
    XYOnionSplit(; n_layers = 5);
    target = y50,
    train = 70,
    test = 30,
    rng = MersenneTwister(1),
  )

  for res in (res1, res5)
    @test length(res.train) + length(res.test) == 50
    @test Set(vcat(res.train, res.test)) == Set(1:50)
  end
end

@testset "XYOnion requires target" begin
  @test_throws SplitInputError partition(X50, XYOnionSplit(); train = 70, test = 30)
end

@testset "XYOnion vector-of-vectors input" begin
  Xvov = [X50[:, i] for i = 1:50]
  res = partition(
    Xvov,
    XYOnionSplit();
    target = y50,
    train = 70,
    test = 30,
    rng = MersenneTwister(1),
  )
  @test length(res.train) + length(res.test) == 50
  @test Set(vcat(res.train, res.test)) == Set(1:50)
end

@testset "XYOnion nosamps > F (SPXY extension path)" begin
  # 3 features, 50 samples → ncalloop can exceed F=3
  X3 = randn(MersenneTwister(7), 3, 50)
  y3 = randn(MersenneTwister(8), 50)
  res = partition(
    X3,
    XYOnionSplit(; n_layers = 5);
    target = y3,
    train = 70,
    test = 30,
    rng = MersenneTwister(1),
  )
  @test length(res.train) + length(res.test) == 50
  @test Set(vcat(res.train, res.test)) == Set(1:50)
end
