using BenchmarkTools
using DataSplits
using Random

Random.seed!(42)

const SUITE = BenchmarkGroup()

# ── Fixtures ─────────────────────────────────────────────────────────────────

function _mat(N, D = 20)
  randn(Float64, D, N)
end

function _groups(N, n_groups = 20)
  repeat(1:n_groups, ceil(Int, N / n_groups))[1:N]
end

# ── Distance-based (O(N²)) ────────────────────────────────────────────────────
# Benchmarked at N = 200 / 500 / 1000 to capture the quadratic scaling.

SUITE["distance"] = BenchmarkGroup()

for N in [200, 500, 1000]
  tag = "N=$N"
  X = _mat(N)
  y = randn(N)

  SUITE["distance"]["KennardStoneSplit"][tag] =
    @benchmarkable partition($X, KennardStoneSplit(); train = 0.8, test = 0.2)
  SUITE["distance"]["LazyKennardStoneSplit"][tag] =
    @benchmarkable partition($X, LazyKennardStoneSplit(); train = 0.8, test = 0.2)
  SUITE["distance"]["MoraisLimaMartinSplit"][tag] =
    @benchmarkable partition($X, MoraisLimaMartinSplit(); train = 0.8, test = 0.2)
  SUITE["distance"]["SPXYSplit"][tag] =
    @benchmarkable partition($X, SPXYSplit(); target = $y, train = 0.8, test = 0.2)
  SUITE["distance"]["LazySPXYSplit"][tag] =
    @benchmarkable partition($X, LazySPXYSplit(); target = $y, train = 0.8, test = 0.2)
  SUITE["distance"]["MDKSSplit"][tag] =
    @benchmarkable partition($X, MDKSSplit(); target = $y, train = 0.8, test = 0.2)
  SUITE["distance"]["LazyMDKSSplit"][tag] =
    @benchmarkable partition($X, LazyMDKSSplit(); target = $y, train = 0.8, test = 0.2)
  SUITE["distance"]["OptiSimSplit"][tag] =
    @benchmarkable partition($X, OptiSimSplit(); train = 0.8, test = 0.2)
  SUITE["distance"]["LazyOptiSimSplit"][tag] =
    @benchmarkable partition($X, LazyOptiSimSplit(); train = 0.8, test = 0.2)
  SUITE["distance"]["MinimumDissimilaritySplit"][tag] =
    @benchmarkable partition($X, MinimumDissimilaritySplit(); train = 0.8, test = 0.2)
  SUITE["distance"]["LazyMinimumDissimilaritySplit"][tag] =
    @benchmarkable partition($X, LazyMinimumDissimilaritySplit(); train = 0.8, test = 0.2)
  SUITE["distance"]["MaximumDissimilaritySplit"][tag] =
    @benchmarkable partition($X, MaximumDissimilaritySplit(); train = 0.8, test = 0.2)
  SUITE["distance"]["LazyMaximumDissimilaritySplit"][tag] =
    @benchmarkable partition($X, LazyMaximumDissimilaritySplit(); train = 0.8, test = 0.2)
end

# ── Simple train/test splits (O(N)) ──────────────────────────────────────────

SUITE["simple"] = BenchmarkGroup()

for N in [1000, 5000]
  tag = "N=$N"
  X = _mat(N)
  y = randn(N)
  groups = _groups(N)
  times = collect(1:N)

  SUITE["simple"]["RandomSplit"][tag] =
    @benchmarkable partition($X, RandomSplit(); train = 0.8, test = 0.2)
  SUITE["simple"]["TargetPropertyHigh"][tag] =
    @benchmarkable partition($X, TargetPropertyHigh(); target = $y, train = 0.8, test = 0.2)
  SUITE["simple"]["TargetPropertyLow"][tag] =
    @benchmarkable partition($X, TargetPropertyLow(); target = $y, train = 0.8, test = 0.2)
  SUITE["simple"]["GroupShuffleSplit"][tag] = @benchmarkable partition(
    $X,
    GroupShuffleSplit();
    groups = $groups,
    train = 0.8,
    test = 0.2,
  )
  SUITE["simple"]["GroupStratifiedSplit"][tag] = @benchmarkable partition(
    $X,
    GroupStratifiedSplit(:proportional);
    groups = $groups,
    train = 0.8,
    test = 0.2,
  )
  SUITE["simple"]["TimeSplitOldest"][tag] =
    @benchmarkable partition($times, TimeSplitOldest(); train = 0.8, test = 0.2)
  SUITE["simple"]["TimeSplitNewest"][tag] =
    @benchmarkable partition($times, TimeSplitNewest(); train = 0.8, test = 0.2)
end

# ── Cross-validation ──────────────────────────────────────────────────────────

SUITE["crossval"] = BenchmarkGroup()

for N in [1000, 5000]
  tag = "N=$N"
  X = _mat(N)
  labels = rand(1:3, N)
  groups = _groups(N)
  folds = mod.(0:(N-1), 5) .+ 1

  SUITE["crossval"]["KFold"][tag] = @benchmarkable partition($X, KFold(5))
  SUITE["crossval"]["StratifiedKFold"][tag] =
    @benchmarkable partition($X, StratifiedKFold(5); target = $labels)
  SUITE["crossval"]["GroupKFold"][tag] =
    @benchmarkable partition($X, GroupKFold(5); groups = $groups)
  SUITE["crossval"]["StratifiedGroupKFold"][tag] = @benchmarkable partition(
    $X,
    StratifiedGroupKFold(5);
    groups = $groups,
    target = $labels,
  )
  SUITE["crossval"]["ShuffleSplit"][tag] =
    @benchmarkable partition($X, ShuffleSplit(10); train = 0.8, test = 0.2)
  SUITE["crossval"]["StratifiedShuffleSplit"][tag] = @benchmarkable partition(
    $X,
    StratifiedShuffleSplit(10);
    target = $labels,
    train = 0.8,
    test = 0.2,
  )
  SUITE["crossval"]["GroupShuffleSplitCV"][tag] = @benchmarkable partition(
    $X,
    GroupShuffleSplitCV(10);
    groups = $groups,
    train = 0.8,
    test = 0.2,
  )
  SUITE["crossval"]["BootstrapSplit"][tag] =
    @benchmarkable partition($X, BootstrapSplit(10))
  SUITE["crossval"]["RepeatedKFold"][tag] =
    @benchmarkable partition($X, RepeatedKFold(5; n_repeats = 3))
  SUITE["crossval"]["RepeatedStratifiedKFold"][tag] = @benchmarkable partition(
    $X,
    RepeatedStratifiedKFold(5; n_repeats = 3);
    target = $labels,
  )
  SUITE["crossval"]["PredefinedSplit"][tag] =
    @benchmarkable partition($X, PredefinedSplit($folds))
end

# ── Time-series cross-validation ──────────────────────────────────────────────

SUITE["time_cv"] = BenchmarkGroup()

for N in [1000, 5000]
  tag = "N=$N"
  X = _mat(N)
  times = collect(1:N)

  SUITE["time_cv"]["TimeSeriesSplit"][tag] =
    @benchmarkable partition($X, TimeSeriesSplit(5); time = $times)
  SUITE["time_cv"]["BlockedCV"][tag] =
    @benchmarkable partition($X, BlockedCV(5); time = $times)
  SUITE["time_cv"]["PurgedKFold"][tag] =
    @benchmarkable partition($X, PurgedKFold(5); time = $times)
end

# ── Leave-p-out ───────────────────────────────────────────────────────────────
# Kept at small N: LeavePOut(2) produces C(N,2) folds.

SUITE["leave_p_out"] = BenchmarkGroup()

for N in [50, 100]
  tag = "N=$N"
  X = _mat(N)
  groups = _groups(N, 10)

  SUITE["leave_p_out"]["LeaveOneOut"][tag] = @benchmarkable partition($X, LeaveOneOut())
  SUITE["leave_p_out"]["LeavePOut"][tag] = @benchmarkable partition($X, LeavePOut(2))
  SUITE["leave_p_out"]["LeaveOneGroupOut"][tag] =
    @benchmarkable partition($X, LeaveOneGroupOut(); groups = $groups)
  SUITE["leave_p_out"]["LeavePGroupsOut"][tag] =
    @benchmarkable partition($X, LeavePGroupsOut(2); groups = $groups)
end

# ── Nested CV ─────────────────────────────────────────────────────────────────

SUITE["nested_cv"] = BenchmarkGroup()

for N in [200, 500]
  tag = "N=$N"
  X = _mat(N)
  labels = rand(1:3, N)
  groups = _groups(N)

  SUITE["nested_cv"]["NestedCV_KFold"][tag] =
    @benchmarkable partition($X, NestedCV(KFold(5), KFold(3)))
  SUITE["nested_cv"]["NestedCV_Stratified"][tag] = @benchmarkable partition(
    $X,
    NestedCV(StratifiedKFold(5), StratifiedKFold(3));
    target = $labels,
  )
  SUITE["nested_cv"]["NestedCV_Group"][tag] =
    @benchmarkable partition($X, NestedCV(GroupKFold(5), GroupKFold(3)); groups = $groups)
end
