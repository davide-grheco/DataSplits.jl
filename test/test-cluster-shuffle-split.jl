using Test, Random, DataSplits, Clustering, Distances
using OffsetArrays
using ArrayInterface
using MLUtils

Random.seed!(42)
X = vcat(randn(30, 2), randn(40, 2) .+ 5, randn(50, 2) .+ 10)'
N = 120

@testset "ClusterShuffleSplit with kmeans" begin
  # kmeans: observations in columns
  res_k = kmeans(X, 3)
  splitter_k = ClusterShuffleSplit(res_k, 0.5)
  result_k = DataSplits.split(X, splitter_k; rng = MersenneTwister(123))
  train_k, test_k = result_k.train, result_k.test
  @test length(train_k) + length(test_k) == N
  @test abs(length(train_k) / N - 0.5) < 0.2

  # On-the-fly kmeans
  f_k = x -> kmeans(x, 3)
  splitter_fk = ClusterShuffleSplit(f_k, 0.5, X)
  result_fk = DataSplits.split(X, splitter_fk; rng = MersenneTwister(123))
  train_fk, test_fk = result_fk.train, result_fk.test
  @test sort(train_fk) == sort(train_k)
  @test sort(test_fk) == sort(test_k)

  # OffsetArray test
  Xoff = OffsetArray(randn(2, 20), 1:2, -10:9)
  res_k_off = kmeans(ArrayInterface.parent(Xoff), 3)
  splitter_k_off = ClusterShuffleSplit(res_k_off, 0.5)
  indices = collect(axes(Xoff, 2))
  result_off = DataSplits.split(Xoff, splitter_k_off; rng = MersenneTwister(123))
  train_off, test_off = result_off.train, result_off.test
  all_indices = sort(vcat(train_off, test_off))
  @test all_indices == collect(1:numobs(Xoff))
  @test isempty(intersect(train_off, test_off))
end

@testset "ClusterShuffleSplit with K-Means" begin
  res_k = kmeans(X, 3)
  @test isa(res_k, ClusteringResult)
  @test nclusters(res_k) == 3
  @test sum(counts(res_k)) == N

  # Precomputed split
  splitter_k = ClusterShuffleSplit(res_k, 0.6)
  result_k = DataSplits.split(X, splitter_k; rng = MersenneTwister(123))
  train_k, test_k = result_k.train, result_k.test
  @test length(train_k) + length(test_k) == N
  @test abs(length(train_k) / N - 0.6) < 0.1

  # On-the-fly split
  f_k = x -> kmeans(x, 3)
  splitter_fk = ClusterShuffleSplit(f_k, 0.6, X)
  result_fk = DataSplits.split(X, splitter_fk; rng = MersenneTwister(123))
  train_fk, test_fk = result_fk.train, result_fk.test
  @test train_fk == train_k
  @test test_fk == test_k
end

@testset "ClusterShuffleSplit with DBSCAN" begin
  f_h = x -> dbscan(pairwise(Euclidean(), x; dims = 2), 2, metric = nothing)
  splitter_h = ClusterShuffleSplit(f_h, 0.5, X)
  result_h = DataSplits.split(X, splitter_h; rng = MersenneTwister(321))
  train_h, test_h = result_h.train, result_h.test
  @test length(train_h) + length(test_h) == N
  @test abs(length(train_h) / N - 0.5) < 0.2
end
