using Test, Random, DataSplits, Dates
import DataSplits: SplitParameterError

@testset "BlockedCV" begin
  N = 50
  ts = collect(1:N)
  X = randn(2, N)

  @testset "Basic contract: k folds, train ∪ test == 1:N" begin
    cvs = partition(X, BlockedCV(5); time = ts)
    @test length(folds(cvs)) == 5
    for f in folds(cvs)
      @test isempty(intersect(f.train, f.test))
      @test sort(vcat(f.train, f.test)) == 1:N
    end
  end

  @testset "Train is bidirectional (unlike TimeSeriesSplit)" begin
    cvs = partition(X, BlockedCV(5); time = ts)
    fs = folds(cvs)
    # Middle folds should have train both before and after the test block.
    middle = fs[3]
    train_min, train_max = extrema(ts[middle.train])
    test_min, test_max = extrema(ts[middle.test])
    @test train_min < test_min
    @test train_max > test_max
  end

  @testset "Test cohorts tile the timeline disjointly" begin
    cvs = partition(X, BlockedCV(5); time = ts)
    test_concat = sort(reduce(vcat, [f.test for f in folds(cvs)]))
    @test test_concat == 1:N
  end

  @testset "Remainder distributed across chunks" begin
    # B=23, k=5 → 23÷5=4 r 3, so block sizes 5,5,5,4,4.
    ts23 = collect(1:23)
    data = randn(2, 23)
    cvs = partition(data, BlockedCV(5); time = ts23)
    test_sizes = [length(f.test) for f in folds(cvs)]
    @test test_sizes == [5, 5, 5, 4, 4]
  end

  @testset "gap removes observations on both sides of the test block" begin
    cvs = partition(X, BlockedCV(5; gap = 2); time = ts)
    fs = folds(cvs)
    middle = fs[3]
    test_min, test_max = extrema(ts[middle.test])
    # No train observation lies within gap of the test block.
    for t in ts[middle.train]
      @test t < test_min - 2 || t > test_max + 2
    end
  end

  @testset "Boundary folds: first fold has no left side, last fold no right side" begin
    cvs = partition(X, BlockedCV(5); time = ts)
    fs = folds(cvs)
    @test minimum(ts[fs[1].test]) == 1
    @test maximum(ts[fs[end].test]) == N
    @test maximum(ts[fs[1].train]) > maximum(ts[fs[1].test])  # only right side
    @test minimum(ts[fs[end].train]) < minimum(ts[fs[end].test])  # only left side
  end

  @testset "Atomicity by timestamp" begin
    ts_rep = repeat(1:10, inner = 3)
    data = randn(2, length(ts_rep))
    cvs = partition(data, BlockedCV(2); time = ts_rep)
    for f in folds(cvs)
      train_ts = unique(ts_rep[f.train])
      test_ts = unique(ts_rep[f.test])
      @test isempty(intersect(train_ts, test_ts))
    end
  end

  @testset "Date timestamps" begin
    dates = Date(2024, 1, 1) .+ Day.(0:(N-1))
    cvs = partition(X, BlockedCV(4); time = dates)
    @test length(folds(cvs)) == 4
  end

  @testset "Fallback: time as both data and time" begin
    cvs = partition(ts, BlockedCV(4))
    @test length(folds(cvs)) == 4
  end

  @testset "Parameter validation" begin
    @test_throws SplitParameterError BlockedCV(1)
    @test_throws SplitParameterError BlockedCV(5; gap = -1)
    @test_throws SplitParameterError partition(randn(2, 3), BlockedCV(5); time = [1, 2, 3])
  end

  @testset "gap consuming the train cohort errors" begin
    @test_throws SplitParameterError partition(X, BlockedCV(2; gap = N); time = ts)
  end
end
