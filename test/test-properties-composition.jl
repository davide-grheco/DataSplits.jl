using DataSplits:
  partition,
  splitdata,
  RandomSplit,
  TimeSplitOldest,
  GroupShuffleSplit,
  TargetPropertyHigh,
  trainindices,
  valindices,
  testindices

const comp_train_val_test_sizes_gen =
  @composed function make_comp_train_val_test_sizes(N = Data.Integers(3, 200))
    n_train = Data.produce!(Data.Integers(1, N - 2))
    n_val = Data.produce!(Data.Integers(1, N - n_train - 1))
    n_test = N - n_train - n_val

    return (N, n_train, n_val, n_test)
  end

const comp_two_way_percentage_case_gen =
  @composed function make_comp_two_way_percentage_case(N = Data.Integers(40, 300))
    train_pct = Data.produce!(Data.Integers(5, 95))
    test_pct = 100 - train_pct

    return (N, train_pct, test_pct)
  end

const comp_three_way_percentage_case_gen =
  @composed function make_comp_three_way_percentage_case(N = Data.Integers(40, 300))
    train_pct = Data.produce!(Data.Integers(5, 90))
    val_pct = Data.produce!(Data.Integers(5, 95 - train_pct))
    test_pct = 100 - train_pct - val_pct

    return (N, train_pct, val_pct, test_pct)
  end

const comp_fallback_case_gen =
  @composed function make_comp_fallback_case(N = Data.Integers(3, 200))
    n_train = Data.produce!(Data.Integers(1, N - 1))
    n_test = N - n_train

    return (N, n_train, n_test)
  end

const comp_group_composition_case_gen =
  @composed function make_comp_group_composition_case(N = Data.Integers(3, 120))
    n_train = Data.produce!(Data.Integers(1, N - 2))
    n_val = Data.produce!(Data.Integers(1, N - n_train - 1))
    n_test = N - n_train - n_val

    groups = collect(1:N)

    return (groups, n_train, n_val, n_test)
  end

function comp_is_full_train_test_partition(result, N)
  train = trainindices(result)
  test = testindices(result)

  return isempty(intersect(train, test)) && sort(vcat(train, test)) == collect(1:N)
end

function comp_is_full_train_val_test_partition(result, N)
  train = trainindices(result)
  val = valindices(result)
  test = testindices(result)

  return isempty(intersect(train, val)) &&
         isempty(intersect(train, test)) &&
         isempty(intersect(val, test)) &&
         sort(vcat(train, val, test)) == collect(1:N)
end

function comp_has_train_val_test_sizes(result, n_train, n_val, n_test)
  return length(trainindices(result)) == n_train &&
         length(valindices(result)) == n_val &&
         length(testindices(result)) == n_test
end

function comp_same_train_test_indices(a, b)
  return trainindices(a) == trainindices(b) && testindices(a) == testindices(b)
end

function comp_same_train_val_test_indices(a, b)
  return trainindices(a) == trainindices(b) &&
         valindices(a) == valindices(b) &&
         testindices(a) == testindices(b)
end

function comp_no_group_leakage_train_val_test(result, groups)
  train_groups = Set(groups[trainindices(result)])
  val_groups = Set(groups[valindices(result)])
  test_groups = Set(groups[testindices(result)])

  return isempty(intersect(train_groups, val_groups)) &&
         isempty(intersect(train_groups, test_groups)) &&
         isempty(intersect(val_groups, test_groups))
end

@testset "partition composition properties" begin
  @testset "RandomSplit + RandomSplit creates a valid train/val/test partition" begin
    @check max_examples = 300 rng = Xoshiro(30) function random_random_three_way_partition(
      case = comp_train_val_test_sizes_gen,
    )
      N, n_train, n_val, n_test = case
      X = reshape(collect(1:N), 1, N)

      result = partition(
        X,
        RandomSplit(),
        RandomSplit();
        train = n_train,
        validation = n_val,
        test = n_test,
        rng = Xoshiro(42),
      )

      pbt_has_train_val_test_sizes(result, n_train, n_val, n_test) &&
        pbt_is_full_train_val_test_partition(result, N)
    end
  end

  @testset "percentage and fraction sizes are consistent for two-way partition" begin
    @check max_examples = 300 rng = Xoshiro(31) function two_way_percentage_and_fraction_sizes_are_consistent(
      case = comp_two_way_percentage_case_gen,
    )
      N, train_pct, test_pct = case
      X = reshape(collect(1:N), 1, N)

      percentage_result =
        partition(X, RandomSplit(); train = train_pct, test = test_pct, rng = Xoshiro(42))

      fraction_result = partition(
        X,
        RandomSplit();
        train = train_pct / 100,
        test = test_pct / 100,
        rng = Xoshiro(42),
      )

      pbt_is_full_train_test_partition(percentage_result, N) &&
        pbt_is_full_train_test_partition(fraction_result, N) &&
        abs(
          length(trainindices(percentage_result)) - length(trainindices(fraction_result)),
        ) <= 1 &&
        abs(
          length(testindices(percentage_result)) - length(testindices(fraction_result)),
        ) <= 1
    end
  end

  @testset "percentage and fraction sizes are consistent for three-way partition" begin
    @check max_examples = 300 rng = Xoshiro(32) function three_way_percentage_and_fraction_sizes_are_consistent(
      case = comp_three_way_percentage_case_gen,
    )
      N, train_pct, val_pct, test_pct = case
      X = reshape(collect(1:N), 1, N)

      percentage_result = partition(
        X,
        RandomSplit(),
        RandomSplit();
        train = train_pct,
        validation = val_pct,
        test = test_pct,
        rng = Xoshiro(42),
      )

      fraction_result = partition(
        X,
        RandomSplit(),
        RandomSplit();
        train = train_pct / 100,
        validation = val_pct / 100,
        test = test_pct / 100,
        rng = Xoshiro(42),
      )

      pbt_is_full_train_val_test_partition(percentage_result, N) &&
        pbt_is_full_train_val_test_partition(fraction_result, N) &&
        abs(
          length(trainindices(percentage_result)) - length(trainindices(fraction_result)),
        ) <= 1 &&
        abs(length(valindices(percentage_result)) - length(valindices(fraction_result))) <=
        1 &&
        abs(
          length(testindices(percentage_result)) - length(testindices(fraction_result)),
        ) <= 2
    end
  end

  @testset "splitdata matches train/test indices" begin
    @check max_examples = 300 rng = Xoshiro(33) function splitdata_matches_two_way_indices(
      case = comp_fallback_case_gen,
    )
      N, n_train, n_test = case
      X = reshape(collect(1:(3*N)), 3, N)

      result =
        partition(X, RandomSplit(); train = n_train, test = n_test, rng = Xoshiro(42))

      X_train, X_test = splitdata(result, X)

      X_train == X[:, trainindices(result)] && X_test == X[:, testindices(result)]
    end
  end

  @testset "splitdata matches train/val/test indices" begin
    @check max_examples = 300 rng = Xoshiro(34) function splitdata_matches_three_way_indices(
      case = comp_train_val_test_sizes_gen,
    )
      N, n_train, n_val, n_test = case
      X = reshape(collect(1:(3*N)), 3, N)

      result = partition(
        X,
        RandomSplit(),
        RandomSplit();
        train = n_train,
        validation = n_val,
        test = n_test,
        rng = Xoshiro(42),
      )

      X_train, X_val, X_test = splitdata(result, X)

      X_train == X[:, trainindices(result)] &&
        X_val == X[:, valindices(result)] &&
        X_test == X[:, testindices(result)]
    end
  end

  @testset "fallback and explicit time= are equivalent" begin
    @check max_examples = 300 rng = Xoshiro(35) function time_fallback_matches_explicit_time(
      case = comp_fallback_case_gen,
    )
      N, n_train, n_test = case
      times = Date(2020, 1, 1) .+ Day.(0:(N-1))
      X = reshape(collect(1:N), 1, N)

      fallback_result = partition(times, TimeSplitOldest(); train = n_train, test = n_test)

      explicit_result =
        partition(X, TimeSplitOldest(); time = times, train = n_train, test = n_test)

      comp_same_train_test_indices(fallback_result, explicit_result)
    end
  end

  @testset "fallback and explicit target= are equivalent" begin
    @check max_examples = 300 rng = Xoshiro(36) function target_fallback_matches_explicit_target(
      case = comp_fallback_case_gen,
    )
      N, n_train, n_test = case

      # Strictly ordered target avoids ambiguity from ties.
      y = collect(1:N)
      X = reshape(collect(1:N), 1, N)

      fallback_result = partition(y, TargetPropertyHigh(); train = n_train, test = n_test)

      explicit_result =
        partition(X, TargetPropertyHigh(); target = y, train = n_train, test = n_test)

      comp_same_train_test_indices(fallback_result, explicit_result)
    end
  end

  @testset "fallback and explicit groups= are equivalent" begin
    @check max_examples = 300 rng = Xoshiro(37) function group_fallback_matches_explicit_groups(
      case = comp_fallback_case_gen,
    )
      N, n_train, n_test = case

      # Singleton groups make the requested absolute sizes feasible.
      groups = collect(1:N)
      X = reshape(collect(1:N), 1, N)

      fallback_result = partition(
        groups,
        GroupShuffleSplit();
        train = n_train,
        test = n_test,
        rng = Xoshiro(42),
      )

      explicit_result = partition(
        X,
        GroupShuffleSplit();
        groups = groups,
        train = n_train,
        test = n_test,
        rng = Xoshiro(42),
      )

      comp_same_train_test_indices(fallback_result, explicit_result)
    end
  end

  @testset "group-aware train/val/test composition keeps groups in one cohort" begin
    @check max_examples = 300 rng = Xoshiro(38) function group_composition_keeps_groups_whole(
      case = comp_group_composition_case_gen,
    )
      groups, n_train, n_val, n_test = case
      N = length(groups)
      X = reshape(collect(1:N), 1, N)

      result = partition(
        X,
        GroupShuffleSplit(),
        GroupShuffleSplit();
        groups = groups,
        train = n_train,
        validation = n_val,
        test = n_test,
        rng = Xoshiro(42),
      )

      pbt_has_train_val_test_sizes(result, n_train, n_val, n_test) &&
        pbt_is_full_train_val_test_partition(result, N) &&
        comp_no_group_leakage_train_val_test(result, groups)
    end
  end
end
