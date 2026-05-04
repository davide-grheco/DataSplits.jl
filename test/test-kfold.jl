using Test, Random, DataSplits

@testset "KFold basic contract" begin
  X = randn(2, 100)
  cvs = partition(X, KFold(5))

  @test cvs isa CrossValidationSplit
  @test length(cvs) == 5
  @test length(folds(cvs)) == 5

  for fold in cvs
    @test fold isa TrainTestSplit
    @test isempty(intersect(fold.train, fold.test))
    @test sort(vcat(fold.train, fold.test)) == collect(1:100)
  end
end

@testset "KFold fold sizes differ by at most 1" begin
  # N divisible by k
  cvs = partition(randn(2, 100), KFold(5))
  fold_sizes = [length(fold.test) for fold in cvs]
  @test all(==(20), fold_sizes)

  # N not divisible by k
  cvs2 = partition(randn(2, 101), KFold(5))
  fold_sizes2 = [length(fold.test) for fold in cvs2]
  @test sum(fold_sizes2) == 101
  @test maximum(fold_sizes2) - minimum(fold_sizes2) <= 1
end

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
