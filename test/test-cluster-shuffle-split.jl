using Test, Random, DataSplits, Clustering, Distances

Random.seed!(42)
X = vcat(randn(30, 2), randn(40, 2) .+ 5, randn(50, 2) .+ 10)
N = size(X, 1)

@testset "ClusterShuffleSplit with kmeans" begin
  # kmeans: observations in columns
  res_k = kmeans(X', 3)
  splitter_k = ClusterShuffleSplit(res_k, 0.5)
  train_k, test_k = DataSplits.split(X, splitter_k; rng = MersenneTwister(123))
  @test length(train_k) + length(test_k) == N
  @test abs(length(train_k) / N - 0.5) < 0.2

  # On-the-fly kmeans
  f_k = x -> kmeans(x', 3)
  splitter_fk = ClusterShuffleSplit(f_k, 0.5, X)
  train_fk, test_fk = DataSplits.split(X, splitter_fk; rng = MersenneTwister(123))
  @test sort(train_fk) == sort(train_k)
  @test sort(test_fk) == sort(test_k)
end
@testset "ClusterShuffleSplit with K-Means" begin
  res_k = kmeans(X', 3)
  @test isa(res_k, ClusteringResult)
  @test nclusters(res_k) == 3
  @test sum(counts(res_k)) == N

  # Precomputed split
  splitter_k = ClusterShuffleSplit(res_k, 0.6)
  train_k, test_k = DataSplits.split(X, splitter_k; rng = MersenneTwister(123))
  @test length(train_k) + length(test_k) == N
  @test abs(length(train_k) / N - 0.6) < 0.1

  # On-the-fly split
  f_k = x -> kmeans(x', 3)
  splitter_fk = ClusterShuffleSplit(f_k, 0.6, X)
  train_fk, test_fk = DataSplits.split(X, splitter_fk; rng = MersenneTwister(123))
  @test train_fk == train_k
  @test test_fk == test_k
end

@testset "ClusterShuffleSplit with DBSCAN" begin
  f_h = x -> dbscan(pairwise(Euclidean(), x; dims = 1), 2, metric = nothing)
  splitter_h = ClusterShuffleSplit(f_h, 0.5, X)
  train_h, test_h = DataSplits.split(X, splitter_h; rng = MersenneTwister(321))
  @test length(train_h) + length(test_h) == N
  @test abs(length(train_h) / N - 0.5) < 0.2
end
