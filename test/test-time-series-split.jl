using Test, Random, DataSplits, Dates

Random.seed!(42)
N = 60
X = randn(2, N)
timestamps = collect(1:N)

@testset "TimeSeriesSplit basic contract" begin
  cvs = partition(X, TimeSeriesSplit(5); time = timestamps)
  @test cvs isa CrossValidationSplit
  @test length(cvs) == 5

  for fold in cvs
    @test fold isa TrainTestSplit
    @test isempty(intersect(fold.train, fold.test))
  end
end

@testset "TimeSeriesSplit expanding window: train always precedes test" begin
  cvs = partition(X, TimeSeriesSplit(5); time = timestamps)
  for fold in cvs
    @test maximum(timestamps[fold.train]) < minimum(timestamps[fold.test])
  end
end

@testset "TimeSeriesSplit expanding window grows monotonically" begin
  cvs = partition(X, TimeSeriesSplit(5); time = timestamps)
  train_sizes = [length(fold.train) for fold in cvs]
  @test issorted(train_sizes)
end

@testset "TimeSeriesSplit max_train_size caps train cohort" begin
  cvs = partition(X, TimeSeriesSplit(5; max_train_size = 8); time = timestamps)
  for fold in cvs
    @test length(fold.train) <= 8
  end
end

@testset "TimeSeriesSplit max_train_size keeps the most recent observations" begin
  cvs = partition(X, TimeSeriesSplit(5; max_train_size = 8); time = timestamps)
  for fold in cvs
    train_max = maximum(timestamps[fold.train])
    test_min = minimum(timestamps[fold.test])
    @test train_max < test_min
    @test maximum(timestamps[fold.train]) - minimum(timestamps[fold.train]) <= 7
  end
end

@testset "TimeSeriesSplit max_train_size larger than train chunk is a no-op" begin
  cvs_capped = partition(X, TimeSeriesSplit(5; max_train_size = 10_000); time = timestamps)
  cvs_default = partition(X, TimeSeriesSplit(5); time = timestamps)
  for (a, b) in zip(cvs_capped, cvs_default)
    @test a.train == b.train
    @test a.test == b.test
  end
end

@testset "TimeSeriesSplit gap separates train from test" begin
  cvs = partition(X, TimeSeriesSplit(5; gap = 2); time = timestamps)
  for fold in cvs
    gap_obs = minimum(timestamps[fold.test]) - maximum(timestamps[fold.train])
    @test gap_obs > 2
  end
end

@testset "TimeSeriesSplit groups identical timestamps atomically" begin
  ts = repeat(1:10, inner = 3)
  data = randn(2, length(ts))
  cvs = partition(data, TimeSeriesSplit(2); time = ts)
  for fold in cvs
    train_ts = unique(ts[fold.train])
    test_ts = unique(ts[fold.test])
    @test isempty(intersect(train_ts, test_ts))
  end
end

@testset "TimeSeriesSplit with Date timestamps" begin
  dates = Date(2024, 1, 1) .+ Day.(0:(N-1))
  cvs = partition(X, TimeSeriesSplit(3); time = dates)
  @test length(cvs) == 3
  for fold in cvs
    @test maximum(dates[fold.train]) < minimum(dates[fold.test])
  end
end

@testset "TimeSeriesSplit fallback: time as both data and time" begin
  cvs = partition(timestamps, TimeSeriesSplit(4))
  @test length(cvs) == 4
end

@testset "TimeSeriesSplit parameter validation" begin
  @test_throws DataSplits.SplitParameterError partition(
    X,
    TimeSeriesSplit(1);
    time = timestamps,
  )
  @test_throws DataSplits.SplitParameterError partition(
    X,
    TimeSeriesSplit(5; gap = -1);
    time = timestamps,
  )
  @test_throws DataSplits.SplitParameterError partition(
    X,
    TimeSeriesSplit(5; max_train_size = 0);
    time = timestamps,
  )
  # k+1 > N
  small_data = randn(2, 3)
  small_ts = [1, 2, 3]
  @test_throws DataSplits.SplitParameterError partition(
    small_data,
    TimeSeriesSplit(5);
    time = small_ts,
  )
end

@testset "TimeSeriesSplit gap consuming the train cohort errors" begin
  @test_throws DataSplits.SplitParameterError partition(
    X,
    TimeSeriesSplit(5; gap = 100);
    time = timestamps,
  )
end

@testset "TimeSeriesSplit rejects mismatched time length" begin
  short = timestamps[1:40]
  @test_throws DataSplits.SplitInputError partition(X, TimeSeriesSplit(3); time = short)
end

@testset "TimeSeriesSplit gap may trim a block but never leaks across train/test" begin
  ts = repeat(1:12, inner = 3)
  data = randn(2, length(ts))
  cvs = partition(data, TimeSeriesSplit(3; gap = 2); time = ts)
  for fold in cvs
    @test isempty(intersect(unique(ts[fold.train]), unique(ts[fold.test])))
    @test maximum(ts[fold.train]) < minimum(ts[fold.test])
  end
end
