using Test, Random, DataSplits, Distances
using Dates: Date, Day
using MLUtils: numobs
using DataSplits:
  partition,
  splitdata,
  trainindices,
  testindices,
  valindices,
  consumes,
  fallback_from_data,
  TrainTestSplit,
  TrainValTestSplit,
  CrossValidationSplit,
  WithValidation,
  RandomSplit,
  KennardStoneSplit,
  SPXYSplit,
  GroupShuffleSplit,
  TimeSplitOldest,
  TargetPropertyHigh,
  SplitNotImplementedError

N = 60

@testset "Disjoint partition and sizes" begin
  X = rand(3, N)
  rng = MersenneTwister(0)
  res = partition(X, WithValidation(RandomSplit(0.8), RandomSplit(0.75)); rng = rng)

  @test res isa TrainValTestSplit
  train, val, test = trainindices(res), valindices(res), testindices(res)
  @test sort(vcat(train, val, test)) == collect(1:N)
  @test isempty(intersect(train, val))
  @test isempty(intersect(train, test))
  @test isempty(intersect(val, test))

  # Outer 0.8 → 48 train_pool / 12 test
  # Inner 0.75 on the pool → 36 train / 12 val
  @test length(test) == 12
  @test length(val) == 12
  @test length(train) == 36
end

@testset "splitdata returns three subsets in order" begin
  X = rand(3, N)
  res = partition(
    X,
    WithValidation(RandomSplit(0.8), RandomSplit(0.75));
    rng = MersenneTwister(1),
  )
  Xtr, Xv, Xte = splitdata(res, X)
  @test size(Xtr, 2) == length(trainindices(res))
  @test size(Xv, 2) == length(valindices(res))
  @test size(Xte, 2) == length(testindices(res))
end

@testset "Slot union: target consumed by inner strategy" begin
  X = rand(4, N)
  y = rand(N)
  w = WithValidation(RandomSplit(0.8), TargetPropertyHigh(0.75))
  @test consumes(w) == (:target,)

  res = partition(X, w; target = y, rng = MersenneTwister(2))
  @test res isa TrainValTestSplit
  train, val, test = trainindices(res), valindices(res), testindices(res)
  @test sort(vcat(train, val, test)) == collect(1:N)

  # Within the train pool the validation strategy keeps the highest targets
  # in `train` and pushes the lowest into `val`.
  @test minimum(y[train]) >= maximum(y[val])
end

@testset "Group-aware composition: groups never split across cohorts" begin
  groups = repeat(1:12; inner = 5)         # 12 groups × 5 obs = 60
  X = rand(2, N)
  w = WithValidation(GroupShuffleSplit(0.8), GroupShuffleSplit(0.75))
  res = partition(X, w; groups = groups, rng = MersenneTwister(3))

  train, val, test = trainindices(res), valindices(res), testindices(res)
  @test sort(vcat(train, val, test)) == collect(1:N)

  for g in unique(groups)
    idxs = findall(==(g), groups)
    locations = (
      any(i -> i in train, idxs),
      any(i -> i in val, idxs),
      any(i -> i in test, idxs),
    )
    @test count(identity, locations) == 1
  end
end

@testset "Time-aware test split, target-aware validation split" begin
  X = rand(2, N)
  dates = [Date(2020, 1, 1) + Day(i - 1) for i = 1:N]
  y = collect(1.0:N)

  w = WithValidation(TimeSplitOldest(0.8), TargetPropertyHigh(0.75))
  res = partition(X, w; time = dates, target = y, rng = MersenneTwister(4))

  train, val, test = trainindices(res), valindices(res), testindices(res)
  @test sort(vcat(train, val, test)) == collect(1:N)

  # All test observations are strictly newer than every train_pool obs
  pool = vcat(train, val)
  @test maximum(dates[pool]) <= minimum(dates[test])
end

@testset "Distance-based outer composition" begin
  X = rand(5, N)
  res = partition(
    X,
    WithValidation(KennardStoneSplit(0.8), RandomSplit(0.75));
    rng = MersenneTwister(5),
  )
  train, val, test = trainindices(res), valindices(res), testindices(res)
  @test sort(vcat(train, val, test)) == collect(1:N)
  @test length(test) == 12
end

@testset "SPXY (consumes :data and :target) as outer" begin
  X = rand(4, N)
  y = rand(N)
  w = WithValidation(SPXYSplit(0.8), RandomSplit(0.75))
  @test :target in consumes(w)
  @test :data in consumes(w)

  res = partition(X, w; target = y, rng = MersenneTwister(6))
  train, val, test = trainindices(res), valindices(res), testindices(res)
  @test sort(vcat(train, val, test)) == collect(1:N)
end

@testset "Reproducibility under fixed rng" begin
  X = rand(3, N)
  w = WithValidation(KennardStoneSplit(0.8), RandomSplit(0.75))
  r1 = partition(X, w; rng = MersenneTwister(7))
  r2 = partition(X, w; rng = MersenneTwister(7))
  @test trainindices(r1) == trainindices(r2)
  @test valindices(r1) == valindices(r2)
  @test testindices(r1) == testindices(r2)
end

@testset "Slot trait composition" begin
  w_data_only = WithValidation(KennardStoneSplit(0.8), KennardStoneSplit(0.8))
  @test consumes(w_data_only) == (:data,)
  @test fallback_from_data(w_data_only) == ()

  w_groups = WithValidation(GroupShuffleSplit(0.8), GroupShuffleSplit(0.8))
  @test consumes(w_groups) == (:groups,)
  @test :groups in fallback_from_data(w_groups)

  w_mixed = WithValidation(TimeSplitOldest(0.8), TargetPropertyHigh(0.8))
  @test :time in consumes(w_mixed)
  @test :target in consumes(w_mixed)
end

# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------

# A toy strategy whose result is not a TrainTestSplit; used to exercise the
# error path in WithValidation.
struct _NonTrainTestStrategy <: DataSplits.AbstractSplitStrategy end
DataSplits.consumes(::_NonTrainTestStrategy) = ()
DataSplits.fallback_from_data(::_NonTrainTestStrategy) = ()
function DataSplits._partition(data, ::_NonTrainTestStrategy; kwargs...)
  n = numobs(data)
  half = div(n, 2)
  inner = TrainTestSplit(collect(1:half), collect(half+1:n))
  return CrossValidationSplit([inner])
end

@testset "Inner result type validation" begin
  X = rand(2, 20)
  w_outer_bad = WithValidation(_NonTrainTestStrategy(), RandomSplit(0.75))
  @test_throws SplitNotImplementedError partition(X, w_outer_bad; rng = MersenneTwister(0))

  w_inner_bad = WithValidation(RandomSplit(0.8), _NonTrainTestStrategy())
  @test_throws SplitNotImplementedError partition(X, w_inner_bad; rng = MersenneTwister(0))
end
