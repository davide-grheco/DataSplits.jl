using Test, Random, DataSplits

Random.seed!(42)
X = vcat(randn(30, 2), randn(40, 2) .+ 5, randn(50, 2) .+ 10)'
N = 120
groups = vcat(fill(:a, 30), fill(:b, 40), fill(:c, 50))


@testset "GroupKFold with many groups produces near-balanced folds" begin
  many_groups = repeat(1:30, inner = 4)
  data = randn(2, length(many_groups))
  cvs = partition(data, GroupKFold(5); groups = many_groups)
  fold_sizes = [length(fold.test) for fold in cvs]
  @test sum(fold_sizes) == length(many_groups)
  @test maximum(fold_sizes) - minimum(fold_sizes) <= 4
end

@testset "GroupKFold fallback: ids as both data and groups" begin
  ids = vcat(fill(:a, 40), fill(:b, 40), fill(:c, 40))
  cvs = partition(ids, GroupKFold(3))
  @test length(cvs) == 3
  for fold in cvs
    @test is_disjoint(fold)
  end
end

@testset "GroupKFold splitview iterates train/test pairs" begin
  cvs = partition(X, GroupKFold(3); groups = groups)
  pairs = splitview(cvs, X)
  @test length(pairs) == 3
  for (X_train, X_test) in pairs
    @test size(X_train, 2) + size(X_test, 2) == N
  end
end

@testset "GroupKFold splitdata returns concrete subsets" begin
  cvs = partition(X, GroupKFold(3); groups = groups)
  parts = splitdata(cvs, X)
  @test length(parts) == 3
  for (X_train, X_test) in parts
    @test size(X_train, 2) + size(X_test, 2) == N
  end
end

@testset "GroupKFold parameter validation" begin
  @test_throws DataSplits.SplitParameterError partition(X, GroupKFold(1); groups = groups)
  @test_throws DataSplits.SplitParameterError partition(X, GroupKFold(0); groups = groups)
  # k > n_groups
  @test_throws DataSplits.SplitParameterError partition(X, GroupKFold(10); groups = groups)
end

@testset "GroupKFold rejects mismatched groups length" begin
  short_groups = groups[1:50]
  @test_throws DataSplits.SplitInputError partition(X, GroupKFold(3); groups = short_groups)
end

@testset "GroupKFold rejects train/test kwargs" begin
  @test_throws MethodError partition(
    X,
    GroupKFold(3);
    groups = groups,
    train = 80,
    test = 20,
  )
end

@testset "GroupKFold deterministic by default" begin
  many = repeat(1:30, inner = 4)
  data = randn(2, length(many))
  cvs1 = partition(data, GroupKFold(5); groups = many)
  cvs2 = partition(data, GroupKFold(5); groups = many)
  for (a, b) in zip(cvs1, cvs2)
    @test a.test == b.test
    @test a.train == b.train
  end
end

@testset "GroupKFold shuffle=true varies with the rng seed" begin
  many = repeat(1:30, inner = 4)
  data = randn(2, length(many))
  alg = GroupKFold(5; shuffle = true)
  cvs1 = partition(data, alg; groups = many, rng = MersenneTwister(1))
  cvs2 = partition(data, alg; groups = many, rng = MersenneTwister(2))
  any_different = any(zip(cvs1, cvs2)) do (a, b)
    Set(a.test) != Set(b.test)
  end
  @test any_different
end

@testset "GroupKFold shuffle=true reproducible with the same seed" begin
  many = repeat(1:30, inner = 4)
  data = randn(2, length(many))
  alg = GroupKFold(5; shuffle = true)
  cvs1 = partition(data, alg; groups = many, rng = MersenneTwister(42))
  cvs2 = partition(data, alg; groups = many, rng = MersenneTwister(42))
  for (a, b) in zip(cvs1, cvs2)
    @test a.test == b.test
    @test a.train == b.train
  end
end
