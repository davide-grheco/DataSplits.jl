using Test, Random, DataSplits, Dates
import DataSplits: SplitParameterError

@testset "PurgedKFold" begin
  N = 50
  ts = collect(1:N)
  X = randn(2, N)

  @testset "Equivalence with BlockedCV when purge == embargo == 0" begin
    purged = partition(X, PurgedKFold(5); time = ts)
    blocked = partition(X, BlockedCV(5); time = ts)
    for (a, b) in zip(folds(purged), folds(blocked))
      @test sort(a.train) == sort(b.train)
      @test sort(a.test) == sort(b.test)
    end
  end

  @testset "purge removes observations only BEFORE the test block" begin
    cvs = partition(X, PurgedKFold(5; purge = 2); time = ts)
    fs = folds(cvs)
    middle = fs[3]
    test_min, test_max = extrema(ts[middle.test])
    train_ts = ts[middle.train]
    @test all(t -> t < test_min - 2 || t > test_max, train_ts)
    # No embargo on the right side: train resumes immediately after test_max + 1
    @test any(t -> t == test_max + 1, train_ts)
  end

  @testset "embargo removes observations only AFTER the test block" begin
    cvs = partition(X, PurgedKFold(5; embargo = 2); time = ts)
    fs = folds(cvs)
    middle = fs[3]
    test_min, test_max = extrema(ts[middle.test])
    train_ts = ts[middle.train]
    @test all(t -> t < test_min || t > test_max + 2, train_ts)
    # No purge on the left side: train extends right up to test_min - 1
    @test any(t -> t == test_min - 1, train_ts)
  end

  @testset "Asymmetric purge + embargo applied together" begin
    cvs = partition(X, PurgedKFold(5; purge = 3, embargo = 1); time = ts)
    fs = folds(cvs)
    middle = fs[3]
    test_min, test_max = extrema(ts[middle.test])
    train_ts = ts[middle.train]
    @test all(t -> t < test_min - 3 || t > test_max + 1, train_ts)
  end

  @testset "Boundary folds: first has no left side, last no right" begin
    cvs = partition(X, PurgedKFold(5; purge = 2, embargo = 2); time = ts)
    fs = folds(cvs)
    @test minimum(ts[fs[1].test]) == 1
    @test maximum(ts[fs[end].test]) == N
    @test maximum(ts[fs[1].train]) > maximum(ts[fs[1].test])
    @test minimum(ts[fs[end].train]) < minimum(ts[fs[end].test])
  end

  @testset "Parameter validation" begin
    @test_throws SplitParameterError PurgedKFold(1)
    @test_throws SplitParameterError PurgedKFold(5; purge = -1)
    @test_throws SplitParameterError PurgedKFold(5; embargo = -1)
    # k > number of distinct timestamps (data-dependent, still caught in _partition):
    @test_throws SplitParameterError partition(X, PurgedKFold(51); time = ts)
    # Empty train cohort:
    @test_throws SplitParameterError partition(
      X,
      PurgedKFold(2; purge = N, embargo = N);
      time = ts,
    )
  end

  @testset "Fallback: timestamps as both data and time" begin
    cvs = partition(ts, PurgedKFold(5; purge = 1, embargo = 1))
    @test length(folds(cvs)) == 5
    for f in folds(cvs)
      @test isempty(intersect(f.train, f.test))
    end
  end
end
