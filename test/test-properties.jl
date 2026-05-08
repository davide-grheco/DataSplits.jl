using DataSplits:
  partition, RandomSplit, TimeSplitOldest, TimeSplitNewest, GroupShuffleSplit

# ---------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------

const split_abs_sizes_gen =
  @composed function make_split_abs_sizes(N = Data.Integers(2, 200))
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

# ---------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------

@testset "Property tests" begin
  @testset "RandomSplit" begin
    @check max_examples = 500 rng = Xoshiro(1) function randomsplit_is_a_partition(
      case = split_abs_sizes_gen,
    )
      N, n_train, n_test = case
      X = reshape(collect(1:N), 1, N)

      result =
        partition(X, RandomSplit(); train = n_train, test = n_test, rng = Xoshiro(42))

      has_correct_split_size(result, n_train, n_test) && is_full_partition(result, N)
    end
  end

  @testset "TimeSplit with unique timestamps" begin
    @check max_examples = 300 rng = Xoshiro(2) function oldest_puts_train_before_test(
      case = split_abs_sizes_gen,
    )
      N, n_train, n_test = case
      times = Date(2020, 1, 1) .+ Day.(0:(N-1))

      result = partition(times, TimeSplitOldest(); train = n_train, test = n_test)

      has_correct_split_size(result, n_train, n_test) &&
        is_full_partition(result, N) &&
        oldest_train_before_test(result, times)
    end

    @check max_examples = 300 rng = Xoshiro(3) function newest_puts_train_after_test(
      case = split_abs_sizes_gen,
    )
      N, n_train, n_test = case
      times = Date(2020, 1, 1) .+ Day.(0:(N-1))

      result = partition(times, TimeSplitNewest(); train = n_train, test = n_test)

      has_correct_split_size(result, n_train, n_test) &&
        is_full_partition(result, N) &&
        newest_train_after_test(result, times)
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
        oldest_train_before_test(result, times)
    end

    @check max_examples = 300 rng = Xoshiro(16) function newest_respects_time_groups(
      case = split_time_groups_case_gen,
    )
      times, n_train, n_test = case
      N = length(times)

      result = partition(times, TimeSplitNewest(); train = n_train, test = n_test)

      is_full_partition(result, N) &&
        no_time_value_split(result, times) &&
        newest_train_after_test(result, times)
    end
  end

  @testset "GroupShuffleSplit with equal-size groups" begin
    @check max_examples = 300 rng = Xoshiro(4) function group_shuffle_does_not_split_equal_groups(
      n_groups = Data.Integers(2, 30),
      group_size = Data.Integers(1, 20),
    )
      groups = repeat(1:n_groups; inner = group_size)
      N = length(groups)

      # Use an absolute train quota of 1 so the first whole group goes to train
      # and at least one remaining group can go to test.
      result =
        partition(groups, GroupShuffleSplit(); train = 1, test = N - 1, rng = Xoshiro(42))

      is_full_partition(result, N) &&
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
        no_group_leakage(result, groups) &&
        !isempty(trainindices(result)) &&
        !isempty(testindices(result))
    end
  end
end
