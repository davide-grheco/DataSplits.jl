using Test, Random, DataSplits
using Statistics: mean

Random.seed!(42)


@testset "StratifiedKFold regression: quantile binning" begin
  Random.seed!(7)
  y = randn(200)
  X = randn(2, 200)
  cvs = partition(X, StratifiedKFold(5; bins = 4); target = y)

  for fold in cvs
    @test is_disjoint(fold)
    train_mean, test_mean = mean(y[fold.train]), mean(y[fold.test])
    @test abs(train_mean - test_mean) < 0.5
  end
end

@testset "StratifiedKFold fallback: target as both data and target" begin
  labels = repeat(['a', 'b', 'c'], inner = 20)
  cvs = partition(labels, StratifiedKFold(4))
  @test length(cvs) == 4
end

@testset "StratifiedKFold splitview iterates train/test pairs" begin
  labels = vcat(fill(:a, 25), fill(:b, 25))
  X = randn(2, 50)
  cvs = partition(X, StratifiedKFold(5); target = labels)
  for (X_train, X_test) in splitview(cvs, X)
    @test size(X_train, 2) + size(X_test, 2) == 50
  end
end

@testset "StratifiedKFold parameter validation" begin
  labels = vcat(fill(:a, 10), fill(:b, 10))
  X = randn(2, 20)
  @test_throws DataSplits.SplitParameterError partition(
    X,
    StratifiedKFold(1);
    target = labels,
  )
  @test_throws DataSplits.SplitParameterError partition(
    X,
    StratifiedKFold(5; bins = 1);
    target = randn(20),
  )
end

@testset "StratifiedKFold rejects classes with too few members" begin
  labels = vcat(fill(:a, 30), fill(:b, 3))
  X = randn(2, 33)
  @test_throws DataSplits.SplitParameterError partition(
    X,
    StratifiedKFold(5);
    target = labels,
  )
end

@testset "StratifiedKFold rejects mismatched target length" begin
  labels = vcat(fill(:a, 30), fill(:b, 30))
  X = randn(2, 60)
  short = labels[1:30]
  @test_throws DataSplits.SplitInputError partition(X, StratifiedKFold(3); target = short)
end

@testset "StratifiedKFold accepts BitVector / Bool targets" begin
  labels = BitVector(vcat(falses(30), trues(30)))
  X = randn(2, 60)
  cvs = partition(X, StratifiedKFold(5); target = labels)
  for fold in cvs
    @test count(labels[fold.test]) == 6
    @test count(.!labels[fold.test]) == 6
  end
end

@testset "StratifiedKFold deterministic by default" begin
  labels = vcat(fill(:a, 30), fill(:b, 30), fill(:c, 30))
  X = randn(2, 90)
  cvs1 = partition(X, StratifiedKFold(5); target = labels)
  cvs2 = partition(X, StratifiedKFold(5); target = labels)
  for (a, b) in zip(cvs1, cvs2)
    @test a.train == b.train
    @test a.test == b.test
  end
end

@testset "StratifiedKFold shuffle=true varies with rng seed" begin
  labels = vcat(fill(:a, 30), fill(:b, 30), fill(:c, 30))
  X = randn(2, 90)
  alg = StratifiedKFold(5; shuffle = true)
  cvs1 = partition(X, alg; target = labels, rng = MersenneTwister(1))
  cvs2 = partition(X, alg; target = labels, rng = MersenneTwister(2))
  any_different = any(zip(cvs1, cvs2)) do (a, b)
    Set(a.test) != Set(b.test)
  end
  @test any_different
end

@testset "StratifiedKFold shuffle=true reproducible with same seed" begin
  labels = vcat(fill(:a, 30), fill(:b, 30), fill(:c, 30))
  X = randn(2, 90)
  alg = StratifiedKFold(5; shuffle = true)
  cvs1 = partition(X, alg; target = labels, rng = MersenneTwister(42))
  cvs2 = partition(X, alg; target = labels, rng = MersenneTwister(42))
  for (a, b) in zip(cvs1, cvs2)
    @test a.train == b.train
    @test a.test == b.test
  end
end

@testset "StratifiedKFold tolerates dense duplicate-value continuous targets" begin
  Random.seed!(11)
  y = vcat(zeros(80), randn(20))
  X = randn(2, 100)
  cvs = partition(X, StratifiedKFold(5; bins = 4); target = y)
  for fold in cvs
    @test is_full_partition(fold, 100)
  end
end
