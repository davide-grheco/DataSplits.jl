using Test, Random, DataSplits


@testset "KFold deterministic by default" begin
  X = randn(2, 50)
  cvs1 = partition(X, KFold(5))
  cvs2 = partition(X, KFold(5))
  for (a, b) in zip(cvs1, cvs2)
    @test a.train == b.train
    @test a.test == b.test
  end
end

@testset "KFold shuffle=true varies with rng seed" begin
  X = randn(2, 50)
  alg = KFold(5; shuffle = true)
  cvs1 = partition(X, alg; rng = MersenneTwister(1))
  cvs2 = partition(X, alg; rng = MersenneTwister(2))
  any_different = any(zip(cvs1, cvs2)) do (a, b)
    Set(a.test) != Set(b.test)
  end
  @test any_different
end

@testset "KFold shuffle=true reproducible with same seed" begin
  X = randn(2, 50)
  alg = KFold(5; shuffle = true)
  cvs1 = partition(X, alg; rng = MersenneTwister(42))
  cvs2 = partition(X, alg; rng = MersenneTwister(42))
  for (a, b) in zip(cvs1, cvs2)
    @test a.train == b.train
    @test a.test == b.test
  end
end

@testset "KFold splitview returns lazy views" begin
  X = randn(2, 60)
  cvs = partition(X, KFold(3))
  pairs = splitview(cvs, X)
  @test length(pairs) == 3
  for (X_train, X_test) in pairs
    @test size(X_train, 2) + size(X_test, 2) == 60
  end
end

@testset "KFold splitdata returns concrete subsets" begin
  X = randn(2, 60)
  cvs = partition(X, KFold(3))
  parts = splitdata(cvs, X)
  @test length(parts) == 3
  for (X_train, X_test) in parts
    @test size(X_train, 2) + size(X_test, 2) == 60
  end
end

@testset "KFold parameter validation" begin
  X = randn(2, 10)
  @test_throws DataSplits.SplitParameterError partition(X, KFold(1))
  @test_throws DataSplits.SplitParameterError partition(X, KFold(0))
  @test_throws DataSplits.SplitParameterError partition(X, KFold(11))
end

@testset "KFold rejects train/test kwargs" begin
  X = randn(2, 10)
  @test_throws MethodError partition(X, KFold(3); train = 80, test = 20)
end
