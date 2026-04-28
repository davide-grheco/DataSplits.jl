using Test, Random, DataSplits, Clustering, Distances

Random.seed!(42)
X = vcat(randn(30, 2), randn(40, 2) .+ 5, randn(50, 2) .+ 10)'
N = 120

@testset "GroupShuffleSplit with explicit groups" begin
  groups = vcat(fill(1, 30), fill(2, 40), fill(3, 50))

  result = partition(
    X,
    GroupShuffleSplit();
    groups = groups,
    train = 50,
    test = 50,
    rng = MersenneTwister(123),
  )
  train, test = result.train, result.test
  @test length(train) + length(test) == N
  @test abs(length(train) / N - 0.5) < 0.2
  @test isempty(intersect(train, test))

  # No group is split between train and test
  for gid in unique(groups)
    idxs = findall(==(gid), groups)
    in_train = count(i -> i in train, idxs)
    in_test = count(i -> i in test, idxs)
    @test in_train == 0 || in_test == 0
  end

  result2 = partition(
    X,
    GroupShuffleSplit();
    groups = groups,
    train = 60,
    test = 40,
    rng = MersenneTwister(123),
  )
  train2, test2 = result2.train, result2.test
  @test length(train2) + length(test2) == N
  @test abs(length(train2) / N - 0.6) < 0.2
end

@testset "GroupShuffleSplit fallback: ids as both data and groups" begin
  ids = vcat(fill(:a, 40), fill(:b, 40), fill(:c, 40))
  result = partition(ids, GroupShuffleSplit(); train = 60, test = 40, rng = MersenneTwister(42))
  @test length(result.train) + length(result.test) == 120
  @test isempty(intersect(result.train, result.test))
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
  @test isempty(intersect(result.train, result.test))
end
