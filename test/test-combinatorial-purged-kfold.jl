using Test, Random, DataSplits, Dates, Combinatorics
import DataSplits: SplitParameterError

@testset "CombinatorialPurgedKFold fold count" begin
  N = 60
  ts = collect(1:N)
  X = randn(2, N)

  for (k, ntf) in [(4, 1), (4, 2), (4, 3), (6, 2), (6, 3)]
    cvs = partition(X, CombinatorialPurgedKFold(k, ntf); time = ts)
    @test length(folds(cvs)) == binomial(k, ntf)
  end
end

@testset "CombinatorialPurgedKFold n_test_folds=1 equivalent to PurgedKFold" begin
  N = 50
  ts = collect(1:N)
  X = randn(2, N)

  cpcv = partition(X, CombinatorialPurgedKFold(5, 1); time = ts)
  pkf = partition(X, PurgedKFold(5); time = ts)

  cpcv_folds = sort([(sort(f.train), sort(f.test)) for f in folds(cpcv)])
  pkf_folds = sort([(sort(f.train), sort(f.test)) for f in folds(pkf)])
  @test cpcv_folds == pkf_folds
end

@testset "CombinatorialPurgedKFold n_test_folds=1 with purge/embargo equivalent to PurgedKFold" begin
  N = 50
  ts = collect(1:N)
  X = randn(2, N)

  cpcv = partition(X, CombinatorialPurgedKFold(5, 1; purge = 2, embargo = 1); time = ts)
  pkf = partition(X, PurgedKFold(5; purge = 2, embargo = 1); time = ts)

  cpcv_folds = sort([(sort(f.train), sort(f.test)) for f in folds(cpcv)])
  pkf_folds = sort([(sort(f.train), sort(f.test)) for f in folds(pkf)])
  @test cpcv_folds == pkf_folds
end

@testset "CombinatorialPurgedKFold each fold is a valid partition" begin
  N = 60
  ts = collect(1:N)
  X = randn(2, N)
  cvs = partition(X, CombinatorialPurgedKFold(6, 2; purge = 1, embargo = 1); time = ts)

  for f in folds(cvs)
    @test isempty(intersect(f.train, f.test))
    @test !isempty(f.train)
    @test !isempty(f.test)
    @test all(i -> 1 <= i <= N, f.train)
    @test all(i -> 1 <= i <= N, f.test)
  end
end

@testset "CombinatorialPurgedKFold purge/embargo respect time order" begin
  N = 60
  ts = collect(1:N)
  X = randn(2, N)

  cvs = partition(X, CombinatorialPurgedKFold(6, 1; purge = 3, embargo = 2); time = ts)

  for f in folds(cvs)
    test_min, test_max = extrema(ts[f.test])
    train_ts = ts[f.train]
    @test all(t -> t < test_min - 3 || t > test_max + 2, train_ts)
  end
end

@testset "CombinatorialPurgedKFold fallback: timestamps as data" begin
  ts = collect(1:40)
  cvs = partition(ts, CombinatorialPurgedKFold(4, 2))
  @test length(folds(cvs)) == binomial(4, 2)
  for f in folds(cvs)
    @test isempty(intersect(f.train, f.test))
  end
end

@testset "CombinatorialPurgedKFold parameter validation" begin
  @test_throws SplitParameterError CombinatorialPurgedKFold(1, 1)
  @test_throws SplitParameterError CombinatorialPurgedKFold(3, 0)
  @test_throws SplitParameterError CombinatorialPurgedKFold(3, 3)
  @test_throws SplitParameterError CombinatorialPurgedKFold(3, 1; purge = -1)
  @test_throws SplitParameterError CombinatorialPurgedKFold(3, 1; embargo = -1)

  N = 10
  ts = collect(1:N)
  X = randn(2, N)
  @test_throws SplitParameterError partition(X, CombinatorialPurgedKFold(11, 1); time = ts)
end
