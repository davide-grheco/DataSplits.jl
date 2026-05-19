using Dates
import DataSplits: SplitParameterError

# Generator: n_times distinct integer timestamps, each repeated 1–5 times.
# k ≤ n_times – 1 so TimeSeriesSplit (needs k+1 non-empty chunks) and
# BlockedCV / PurgedKFold (need k chunks) both accept the same inputs.
const time_cv_gen = @composed function make_time_cv_case(n_times = Data.Integers(3, 20))
  times = Int[]
  for t = 1:n_times
    reps = Data.produce!(Data.Integers(1, 5))
    append!(times, fill(t, reps))
  end
  k = Data.produce!(Data.Integers(2, max(2, n_times - 1)))
  return (times, k)
end

# Separate generator for PurgedKFold with purge/embargo: larger blocks ensure
# train is never fully consumed (≥ 3 distinct timestamps per block).
const purged_gap_gen =
  @composed function make_purged_gap_case(n_times = Data.Integers(6, 25))
    times = Int[]
    for t = 1:n_times
      reps = Data.produce!(Data.Integers(1, 4))
      append!(times, fill(t, reps))
    end
    k = Data.produce!(Data.Integers(2, max(2, n_times ÷ 3)))
    return (times, k)
  end

# Helper: in each fold, every training timestamp precedes every test timestamp.
function train_precedes_test(fold, times)
  tr = times[trainindices(fold)]
  te = times[testindices(fold)]
  isempty(tr) || maximum(tr) < minimum(te)
end

# ------------------------------------------------------------------
# TimeSeriesSplit
# Properties: correct number of folds, every observation tests once,
# train always before test in time, no timestamp value split across cohorts.
# ------------------------------------------------------------------

@testset "TimeSeriesSplit properties" begin
  @check max_examples = 200 rng = Xoshiro(80) function tss_temporal_order_and_coverage(
    case = time_cv_gen,
  )
    times, k = case
    N = length(times)
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, TimeSeriesSplit(k); time = times)
    # Test cohorts are pairwise disjoint (first chunk is only ever in training).
    test_idx_sets = [Set(testindices(f)) for f in cvs]
    pairwise_disjoint = all(
      isempty(intersect(test_idx_sets[i], test_idx_sets[j])) for
      i in 1:k, j in 1:k if i < j
    )
    length(cvs) == k &&
      pairwise_disjoint &&
      all(train_precedes_test(fold, times) for fold in cvs) &&
      all(no_time_value_split(fold, times) for fold in cvs)
  end
end

# ------------------------------------------------------------------
# BlockedCV
# Properties: correct number of folds, all observations in every fold
# (train ∪ test = 1:N), every observation tests once, no timestamp split.
# ------------------------------------------------------------------

@testset "BlockedCV properties" begin
  @check max_examples = 200 rng = Xoshiro(81) function blockedcv_covers_all_obs(
    case = time_cv_gen,
  )
    times, k = case
    N = length(times)
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, BlockedCV(k); time = times)
    length(cvs) == k &&
      is_full_partition(cvs, N) &&
      every_observation_tests_once(cvs, N) &&
      all(no_time_value_split(fold, times) for fold in cvs)
  end
end

# ------------------------------------------------------------------
# PurgedKFold
# Without purge/embargo the result is identical to BlockedCV.
# With purge/embargo, only the training cohort shrinks — the test cohorts
# still tile 1:N exactly and timestamps remain atomic.
# ------------------------------------------------------------------

@testset "PurgedKFold properties (no purge/embargo)" begin
  @check max_examples = 200 rng = Xoshiro(82) function purged_no_gap_covers_all(
    case = time_cv_gen,
  )
    times, k = case
    N = length(times)
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, PurgedKFold(k); time = times)
    length(cvs) == k &&
      is_full_partition(cvs, N) &&
      every_observation_tests_once(cvs, N) &&
      all(no_time_value_split(fold, times) for fold in cvs)
  end
end

@testset "PurgedKFold properties (purge=1, embargo=1)" begin
  @check max_examples = 150 rng = Xoshiro(83) function purged_test_tiles_with_gaps(
    case = purged_gap_gen,
  )
    times, k = case
    N = length(times)
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, PurgedKFold(k; purge = 1, embargo = 1); time = times)
    # Purge/embargo only affect training — test cohorts still tile 1:N exactly.
    every_observation_tests_once(cvs, N) &&
      all(no_time_value_split(fold, times) for fold in cvs)
  end
end
