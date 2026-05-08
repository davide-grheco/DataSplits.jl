using Test, Random, DataSplits, Clustering, Distances

Random.seed!(42)
X = vcat(randn(30, 2), randn(40, 2) .+ 5, randn(50, 2) .+ 10)'
N = 120


@testset "GroupShuffleSplit fallback: ids as both data and groups" begin
  ids = vcat(fill(:a, 40), fill(:b, 40), fill(:c, 40))
  result =
    partition(ids, GroupShuffleSplit(); train = 60, test = 40, rng = MersenneTwister(42))
  @test length(result.train) + length(result.test) == 120
  @test is_disjoint(result)
end

@testset "GroupShuffleSplit with Clustering.jl assignments" begin
  res_k = kmeans(X, 3)
  result = partition(
    X,
    GroupShuffleSplit();
    groups = assignments(res_k),
    train = 70,
    test = 30,
    rng = MersenneTwister(1),
  )
  @test length(result.train) + length(result.test) == N
  @test is_disjoint(result)
end
