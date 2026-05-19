using Test, Random, DataSplits, Dates
import DataSplits: SplitParameterError

@testset "PurgedKFold" begin
  N = 50
  ts = collect(1:N)
  X = randn(2, N)

  @testset "Basic contract: k folds, train ∪ test == 1:N (no purge/embargo)" begin
    cvs = partition(X, PurgedKFold(5); time = ts)
    @test length(folds(cvs)) == 5
    for f in folds(cvs)
      @test isempty(intersect(f.train, f.test))
      @test sort(vcat(f.train, f.test)) == 1:N
    end
  end

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

  @testset "Test cohorts tile the timeline disjointly" begin
    cvs = partition(X, PurgedKFold(5; purge = 2, embargo = 2); time = ts)
    test_concat = sort(reduce(vcat, [f.test for f in folds(cvs)]))
    @test test_concat == 1:N
  end

  @testset "Boundary folds: first has no left side, last no right" begin
    cvs = partition(X, PurgedKFold(5; purge = 2, embargo = 2); time = ts)
    fs = folds(cvs)
    @test minimum(ts[fs[1].test]) == 1
    @test maximum(ts[fs[end].test]) == N
    @test maximum(ts[fs[1].train]) > maximum(ts[fs[1].test])
    @test minimum(ts[fs[end].train]) < minimum(ts[fs[end].test])
  end

  @testset "Atomicity by timestamp (ties never split across train/test)" begin
    ts_dup = repeat(1:10, inner = 5)  # 50 obs, 10 distinct times, 5 ties each.
    Xd = randn(2, 50)
    cvs = partition(Xd, PurgedKFold(5); time = ts_dup)
    for f in folds(cvs)
      train_times = Set(ts_dup[f.train])
      test_times = Set(ts_dup[f.test])
      @test isempty(intersect(train_times, test_times))
    end
  end

  @testset "Parameter validation" begin
    @test_throws SplitParameterError partition(X, PurgedKFold(1); time = ts)
    @test_throws SplitParameterError partition(X, PurgedKFold(5; purge = -1); time = ts)
    @test_throws SplitParameterError partition(X, PurgedKFold(5; embargo = -1); time = ts)
    # k > number of distinct timestamps:
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
