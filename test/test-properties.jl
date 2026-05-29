using Dates

# Generators

const split_abs_sizes_gen =
  @composed function make_split_abs_sizes(N = Data.Integers(2, 200))
    n_train = Data.produce!(Data.Integers(1, N - 1))
    return (N, n_train, N - n_train)
  end

const algo_sizes_gen = @composed function make_algo_sizes(N = Data.Integers(3, 50))
  n_train = Data.produce!(Data.Integers(1, N - 1))
  return (N, n_train, N - n_train)
end

const split_time_groups_case_gen =
  @composed function make_split_time_groups_case(n_times = Data.Integers(2, 50))
    times = Date[]
    for t = 1:n_times
      reps = Data.produce!(Data.Integers(1, 8))
      append!(times, fill(Date(2020, 1, 1) + Day(t - 1), reps))
    end
    N = length(times)
    n_train = Data.produce!(Data.Integers(1, N - 1))
    return (times, n_train, N - n_train)
  end

const split_grouped_case_gen =
  @composed function make_split_grouped_case(n_groups = Data.Integers(2, 30))
    groups = Int[]
    for g = 1:n_groups
      group_size = Data.produce!(Data.Integers(1, 12))
      append!(groups, fill(g, group_size))
    end
    return groups
  end

# ------------------------------------------------------------------
# X-only split strategies
# All share: is_full_partition + cohorts_are_complements + has_correct_split_size.
# ------------------------------------------------------------------

const XONLY_SPLITS = [
  (
    "RandomSplit",
    (X, tr, te) -> partition(X, RandomSplit(); train = tr, test = te, rng = Xoshiro(42)),
  ),
  (
    "KennardStoneSplit",
    (X, tr, te) -> partition(X, KennardStoneSplit(); train = tr, test = te),
  ),
  (
    "LazyKennardStoneSplit",
    (X, tr, te) -> partition(X, LazyKennardStoneSplit(); train = tr, test = te),
  ),
  (
    "MoraisLimaMartinSplit",
    (X, tr, te) ->
      partition(X, MoraisLimaMartinSplit(); train = tr, test = te, rng = Xoshiro(42)),
  ),
  (
    "FieldStrengthSplit",
    (X, tr, te) -> partition(X, FieldStrengthSplit(); train = tr, test = te),
  ),
  ("DuplexSplit", (X, tr, te) -> partition(X, DuplexSplit(); train = tr, test = te)),
  (
    "LazyDuplexSplit",
    (X, tr, te) -> partition(X, LazyDuplexSplit(); train = tr, test = te),
  ),
  (
    "OptiSimSplit",
    (X, tr, te) -> partition(
      X,
      OptiSimSplit(; distance_cutoff = 0.0);
      train = tr,
      test = te,
      rng = Xoshiro(42),
    ),
  ),
  (
    "LazyOptiSimSplit",
    (X, tr, te) -> partition(
      X,
      LazyOptiSimSplit(; distance_cutoff = 0.0);
      train = tr,
      test = te,
      rng = Xoshiro(42),
    ),
  ),
  (
    "MaximumDissimilaritySplit",
    (X, tr, te) -> partition(
      X,
      MaximumDissimilaritySplit(; distance_cutoff = 0.0);
      train = tr,
      test = te,
      rng = Xoshiro(42),
    ),
  ),
  (
    "LazyMaximumDissimilaritySplit",
    (X, tr, te) -> partition(
      X,
      LazyMaximumDissimilaritySplit(; distance_cutoff = 0.0);
      train = tr,
      test = te,
      rng = Xoshiro(42),
    ),
  ),
  (
    "MinimumDissimilaritySplit",
    (X, tr, te) -> partition(
      X,
      MinimumDissimilaritySplit(; distance_cutoff = 0.0);
      train = tr,
      test = te,
      rng = Xoshiro(42),
    ),
  ),
  (
    "LazyMinimumDissimilaritySplit",
    (X, tr, te) -> partition(
      X,
      LazyMinimumDissimilaritySplit(; distance_cutoff = 0.0);
      train = tr,
      test = te,
      rng = Xoshiro(42),
    ),
  ),
]

for (label, runner) in XONLY_SPLITS
  @testset "$label partition invariants" begin
    @check max_examples = 200 rng = Xoshiro(1) function xonly_valid_partition(
      case = algo_sizes_gen,
    )
      N, n_train, n_test = case
      X = reshape(collect(1.0:N), 1, N)
      result = runner(X, n_train, n_test)
      is_full_partition(result, N) &&
        cohorts_are_complements(result, N) &&
        has_correct_split_size(result, n_train, n_test)
    end
  end
end

# ------------------------------------------------------------------
# Lazy/eager split agreement
# Same algorithm, different memory layout — must produce identical selections.
# ------------------------------------------------------------------

const LAZY_EAGER_PAIRS = [
  (
    "KennardStone",
    (X, tr, te) -> partition(X, KennardStoneSplit(); train = tr, test = te),
    (X, tr, te) -> partition(X, LazyKennardStoneSplit(); train = tr, test = te),
  ),
  (
    "SPXY",
    (X, tr, te) ->
      partition(X, SPXYSplit(); target = collect(1.0:size(X, 2)), train = tr, test = te),
    (X, tr, te) -> partition(
      X,
      LazySPXYSplit();
      target = collect(1.0:size(X, 2)),
      train = tr,
      test = te,
    ),
  ),
  (
    "OptiSim",
    (X, tr, te) -> partition(X, OptiSimSplit(); train = tr, test = te, rng = Xoshiro(42)),
    (X, tr, te) ->
      partition(X, LazyOptiSimSplit(); train = tr, test = te, rng = Xoshiro(42)),
  ),
  (
    "MaximumDissimilarity",
    (X, tr, te) ->
      partition(X, MaximumDissimilaritySplit(); train = tr, test = te, rng = Xoshiro(42)),
    (X, tr, te) -> partition(
      X,
      LazyMaximumDissimilaritySplit();
      train = tr,
      test = te,
      rng = Xoshiro(42),
    ),
  ),
  (
    "MinimumDissimilarity",
    (X, tr, te) ->
      partition(X, MinimumDissimilaritySplit(); train = tr, test = te, rng = Xoshiro(42)),
    (X, tr, te) -> partition(
      X,
      LazyMinimumDissimilaritySplit();
      train = tr,
      test = te,
      rng = Xoshiro(42),
    ),
  ),
  (
    "Duplex",
    (X, tr, te) -> partition(X, DuplexSplit(); train = tr, test = te),
    (X, tr, te) -> partition(X, LazyDuplexSplit(); train = tr, test = te),
  ),
]

for (label, eager_runner, lazy_runner) in LAZY_EAGER_PAIRS
  @testset "$label lazy/eager agreement" begin
    @check max_examples = 150 rng = Xoshiro(2) function lazy_eager_agree(
      case = algo_sizes_gen,
    )
      N, n_train, n_test = case
      X = reshape(collect(1.0:N), 1, N)
      r_eager = eager_runner(X, n_train, n_test)
      r_lazy = lazy_runner(X, n_train, n_test)
      Set(trainindices(r_eager)) == Set(trainindices(r_lazy))
    end
  end
end

# ------------------------------------------------------------------
# Target-requiring split strategies (X + y)
# SPXY, MDKS and their lazy variants share the same basic invariants.
# ------------------------------------------------------------------

const TARGET_SPLITS = [
  (
    "SPXYSplit",
    (X, y, tr, te) -> partition(X, SPXYSplit(); target = y, train = tr, test = te),
  ),
  (
    "LazySPXYSplit",
    (X, y, tr, te) -> partition(X, LazySPXYSplit(); target = y, train = tr, test = te),
  ),
  (
    "MDKSSplit",
    (X, y, tr, te) -> partition(X, MDKSSplit(); target = y, train = tr, test = te),
  ),
  (
    "LazyMDKSSplit",
    (X, y, tr, te) -> partition(X, LazyMDKSSplit(); target = y, train = tr, test = te),
  ),
]

for (label, runner) in TARGET_SPLITS
  @testset "$label partition invariants" begin
    @check max_examples = 200 rng = Xoshiro(3) function target_valid_partition(
      case = algo_sizes_gen,
    )
      N, n_train, n_test = case
      X = reshape(collect(1.0:N), 1, N)
      y = collect(1.0:N)
      result = runner(X, y, n_train, n_test)
      is_full_partition(result, N) &&
        cohorts_are_complements(result, N) &&
        has_correct_split_size(result, n_train, n_test)
    end
  end
end

# ------------------------------------------------------------------
# OnionSplit — approximate sizes (rounding per layer), only check coverage
# ------------------------------------------------------------------

@testset "OnionSplit partition invariants" begin
  @check max_examples = 200 rng = Xoshiro(7) function onion_valid_partition(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    result = partition(X, OnionSplit(); train = n_train, test = n_test)
    is_full_partition(result, N) && cohorts_are_complements(result, N)
  end
end

@testset "XYOnionSplit partition invariants" begin
  @check max_examples = 200 rng = Xoshiro(8) function xyonion_valid_partition(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    y = collect(1.0:N)
    result = partition(X, XYOnionSplit(); target = y, train = n_train, test = n_test)
    is_full_partition(result, N) && cohorts_are_complements(result, N)
  end
end

# ------------------------------------------------------------------
# TargetProperty — unique direction property (ordering of train vs test)
# ------------------------------------------------------------------

@testset "TargetPropertyHigh selects largest" begin
  @check max_examples = 300 rng = Xoshiro(50) function target_high_selects_largest(
    case = split_abs_sizes_gen,
  )
    N, n_train, n_test = case
    y = collect(1:N)
    result = partition(y, TargetPropertyHigh(); train = n_train, test = n_test)
    is_full_partition(result, N) &&
      cohorts_are_complements(result, N) &&
      has_correct_split_size(result, n_train, n_test) &&
      minimum(y[trainindices(result)]) >= maximum(y[testindices(result)])
  end
end

@testset "TargetPropertyLow selects smallest" begin
  @check max_examples = 300 rng = Xoshiro(51) function target_low_selects_smallest(
    case = split_abs_sizes_gen,
  )
    N, n_train, n_test = case
    y = collect(1:N)
    result = partition(y, TargetPropertyLow(); train = n_train, test = n_test)
    is_full_partition(result, N) &&
      cohorts_are_complements(result, N) &&
      has_correct_split_size(result, n_train, n_test) &&
      maximum(y[trainindices(result)]) <= minimum(y[testindices(result)])
  end
end

# ------------------------------------------------------------------
# TimeSplit — unique temporal ordering properties
# ------------------------------------------------------------------

@testset "TimeSplit with unique timestamps" begin
  @check max_examples = 300 rng = Xoshiro(2) function oldest_puts_train_before_test(
    case = split_abs_sizes_gen,
  )
    N, n_train, n_test = case
    times = Date(2020, 1, 1) .+ Day.(0:(N-1))
    result = partition(times, TimeSplitOldest(); train = n_train, test = n_test)
    has_correct_split_size(result, n_train, n_test) &&
      is_full_partition(result, N) &&
      oldest_train_before_test(result, times) &&
      cohorts_are_complements(result, N)
  end

  @check max_examples = 300 rng = Xoshiro(3) function newest_puts_train_after_test(
    case = split_abs_sizes_gen,
  )
    N, n_train, n_test = case
    times = Date(2020, 1, 1) .+ Day.(0:(N-1))
    result = partition(times, TimeSplitNewest(); train = n_train, test = n_test)
    has_correct_split_size(result, n_train, n_test) &&
      is_full_partition(result, N) &&
      newest_train_after_test(result, times) &&
      cohorts_are_complements(result, N)
  end
end

@testset "TimeSplit with repeated timestamps" begin
  @check max_examples = 300 rng = Xoshiro(15) function oldest_respects_time_groups(
    case = split_time_groups_case_gen,
  )
    times, n_train, n_test = case
    N = length(times)
    result = partition(times, TimeSplitOldest(); train = n_train, test = n_test)
    is_full_partition(result, N) &&
      no_time_value_split(result, times) &&
      oldest_train_before_test(result, times) &&
      cohorts_are_complements(result, N)
  end

  @check max_examples = 300 rng = Xoshiro(16) function newest_respects_time_groups(
    case = split_time_groups_case_gen,
  )
    times, n_train, n_test = case
    N = length(times)
    result = partition(times, TimeSplitNewest(); train = n_train, test = n_test)
    is_full_partition(result, N) &&
      no_time_value_split(result, times) &&
      newest_train_after_test(result, times) &&
      cohorts_are_complements(result, N)
  end
end

# ------------------------------------------------------------------
# GroupShuffleSplit — unique group non-leakage property
# ------------------------------------------------------------------

@testset "GroupShuffleSplit with equal-size groups" begin
  @check max_examples = 300 rng = Xoshiro(4) function group_shuffle_does_not_split_equal_groups(
    n_groups = Data.Integers(2, 30),
    group_size = Data.Integers(1, 20),
  )
    groups = repeat(1:n_groups; inner = group_size)
    N = length(groups)
    result =
      partition(groups, GroupShuffleSplit(); train = 1, test = N - 1, rng = Xoshiro(42))
    is_full_partition(result, N) &&
      cohorts_are_complements(result, N) &&
      no_group_leakage(result, groups) &&
      !isempty(trainindices(result)) &&
      !isempty(testindices(result))
  end
end

@testset "GroupShuffleSplit with variable-size groups" begin
  @check max_examples = 300 rng = Xoshiro(17) function group_shuffle_does_not_split_variable_groups(
    groups = split_grouped_case_gen,
  )
    N = length(groups)
    X = reshape(collect(1:N), 1, N)
    result = partition(
      X,
      GroupShuffleSplit();
      groups = groups,
      train = 1,
      test = N - 1,
      rng = Xoshiro(42),
    )
    is_full_partition(result, N) &&
      cohorts_are_complements(result, N) &&
      no_group_leakage(result, groups) &&
      !isempty(trainindices(result)) &&
      !isempty(testindices(result))
  end
end

@testset "SpectralSplit partition invariants" begin
  @check max_examples = 100 rng = Xoshiro(9) function spectral_valid_partition(
    case = algo_sizes_gen,
  )
    N, n_train, n_test = case
    X = reshape(collect(1.0:N), 1, N)
    result = partition(
      X,
      SpectralSplit(min(5, N));
      train = n_train,
      test = n_test,
      rng = Xoshiro(42),
    )
    is_full_partition(result, N) && cohorts_are_complements(result, N)
  end
end
