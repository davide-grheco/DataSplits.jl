using Test
using DataSplits: split, ClusterStratifiedSplit
using DataSplits: SplitInputError, SplitParameterError, SplitNotImplementedError
using Clustering

@testset "ClusterStratifiedSplit" begin
  # Simple synthetic data
  X = reshape(1:15, 15, 1) # 3 clusters, 5 samples each, 1 feature
  assigns = vcat(fill(1, 5), fill(2, 5), fill(3, 5))
  struct DummyClusteringResult <: Clustering.ClusteringResult
    assignments::Vector{Int}
  end
  Clustering.assignments(res::DummyClusteringResult) = res.assignments
  clusters = DummyClusteringResult(assigns)

  # Equal allocation: 4 per cluster, 50% train
  s_eq = ClusterStratifiedSplit(clusters, :equal; n = 4, frac = 0.5)
  result = split(X, s_eq)
  train, test = result.train, result.test
  @test length(train) == 6
  @test length(test) == 6
  @test all(count(==(cid), assigns[train]) == 2 for cid = 1:3)
  @test all(count(==(cid), assigns[test]) == 2 for cid = 1:3)

  # Proportional allocation: 50% train
  s_prop = ClusterStratifiedSplit(clusters, :proportional; frac = 0.5)
  result = split(X, s_prop)
  train, test = result.train, result.test
  @test length(train) == 9
  @test length(test) == 6
  @test all(
    count(==(cid), assigns[train]) + count(==(cid), assigns[test]) == 5 for cid = 1:3
  )

  # Neyman allocation: 4 per cluster, 50% train
  s_neyman = ClusterStratifiedSplit(clusters, :neyman; n = 4, frac = 0.5)
  result = split(X, s_neyman)
  train, test = result.train, result.test
  @test length(train) == 6
  @test length(test) == 6
  @test all(
    count(==(cid), assigns[train]) + count(==(cid), assigns[test]) <= 4 for cid = 1:3
  )
end
