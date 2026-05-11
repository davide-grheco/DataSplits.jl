using DataSplits: partition, KFold, trainindices, testindices

const kfold_case_gen = @composed function make_kfold_case(N = Data.Integers(2, 200))
  k = Data.produce!(Data.Integers(2, N))
  return (N, k)
end

@testset "KFold properties" begin
  @check max_examples = 300 rng = Xoshiro(10) function kfold_is_valid_cv(
    case = kfold_case_gen,
  )
    N, k = case
    X = reshape(collect(1:N), 1, N)

    cvs = partition(X, KFold(k))

    length(cvs) == k &&
      all(fold -> is_full_partition(fold, N), cvs) &&
      every_observation_tests_once(cvs, N) &&
      fold_test_sizes_balanced(cvs)
  end
end

const shuffle_split_case_gen = @composed function make_shuffle_split_case(
  N = Data.Integers(2, 100),
  n_splits = Data.Integers(1, 10),
)
  n_train = Data.produce!(Data.Integers(1, N - 1))
  return (N, n_splits, n_train, N - n_train)
end

@testset "ShuffleSplit properties" begin
  @check max_examples = 300 rng = Xoshiro(40) function shuffle_split_valid_resamples(
    case = shuffle_split_case_gen,
  )
    N, n_splits, n_train, n_test = case
    X = reshape(collect(1:N), 1, N)

    cvs = partition(
      X,
      ShuffleSplit(n_splits);
      train = n_train,
      test = n_test,
      rng = Xoshiro(42),
    )

    length(folds(cvs)) == n_splits &&
      all(fold -> is_full_partition(fold, N), folds(cvs)) &&
      all(fold -> has_correct_split_size(fold, n_train, n_test), folds(cvs))
  end
end

const strat_shuffle_case_gen =
  @composed function make_strat_shuffle_case(n_classes = Data.Integers(2, 4))
    n_splits = Data.produce!(Data.Integers(1, 5))
    labels = Symbol[]
    for c = 1:n_classes
      count = Data.produce!(Data.Integers(2, 10))
      append!(labels, fill(Symbol(:c, c), count))
    end
    N = length(labels)
    n_test = Data.produce!(Data.Integers(n_classes, N - n_classes))
    n_train = N - n_test
    return (labels, n_splits, n_train, n_test)
  end

@testset "StratifiedShuffleSplit properties" begin
  @check max_examples = 200 rng = Xoshiro(41) function strat_shuffle_valid_resamples(
    case = strat_shuffle_case_gen,
  )
    labels, n_splits, n_train, n_test = case
    N = length(labels)
    X = reshape(collect(1:N), 1, N)

    cvs = partition(
      X,
      StratifiedShuffleSplit(n_splits);
      target = labels,
      train = n_train,
      test = n_test,
      rng = Xoshiro(42),
    )

    length(folds(cvs)) == n_splits &&
      all(fold -> is_full_partition(fold, N), folds(cvs)) &&
      all(fold -> has_correct_split_size(fold, n_train, n_test), folds(cvs))
  end
end

const grouped_case_gen =
  @composed function make_grouped_case(n_groups = Data.Integers(2, 30))
    groups = Int[]
    for g = 1:n_groups
      group_size = Data.produce!(Data.Integers(1, 12))
      append!(groups, fill(g, group_size))
    end
    k = Data.produce!(Data.Integers(2, n_groups))
    return (groups, k)
  end

@testset "GroupKFold properties" begin
  @check max_examples = 300 rng = Xoshiro(11) function groupkfold_is_valid(
    case = grouped_case_gen,
  )
    groups, k = case
    N = length(groups)
    X = reshape(collect(1:N), 1, N)

    cvs = partition(X, GroupKFold(k); groups = groups)

    length(cvs) == k &&
      all(fold -> is_full_partition(fold, N), cvs) &&
      all(fold -> no_group_leakage(fold, groups), cvs) &&
      every_observation_tests_once(cvs, N) &&
      every_group_tests_once(cvs, groups)
  end
end

const stratified_case_gen =
  @composed function make_stratified_case(n_classes = Data.Integers(2, 8))
    k = Data.produce!(Data.Integers(2, 10))
    labels = Symbol[]
    for c = 1:n_classes
      # Ensure each class has at least k members.
      extra = Data.produce!(Data.Integers(0, 20))
      append!(labels, fill(Symbol(:c, c), k + extra))
    end
    return (labels, k)
  end

@testset "StratifiedKFold properties" begin
  @check max_examples = 300 rng = Xoshiro(12) function stratified_kfold_balances_classes(
    case = stratified_case_gen,
  )
    labels, k = case
    N = length(labels)
    X = reshape(collect(1:N), 1, N)

    cvs = partition(X, StratifiedKFold(k); target = labels)

    length(cvs) == k &&
      all(fold -> is_full_partition(fold, N), cvs) &&
      every_observation_tests_once(cvs, N) &&
      class_counts_balanced_across_test_folds(cvs, labels)
  end
end

const leavepout_case_gen = @composed function make_leavepout_case(N = Data.Integers(2, 10))
  p = Data.produce!(Data.Integers(1, N - 1))
  return (N, p)
end

@testset "LeavePOut properties" begin
  @check max_examples = 150 rng = Xoshiro(13) function leavepout_enumerates_combinations(
    case = leavepout_case_gen,
  )
    N, p = case
    X = reshape(collect(1:N), 1, N)

    cvs = partition(X, LeavePOut(p))
    test_sets = Set(Set(testindices(fold)) for fold in cvs)

    length(cvs) == binomial(N, p) &&
      length(test_sets) == binomial(N, p) &&
      all(fold -> length(testindices(fold)) == p, cvs) &&
      all(fold -> length(trainindices(fold)) == N - p, cvs) &&
      all(fold -> is_full_partition(fold, N), cvs)
  end
end

const leavepgroups_case_gen =
  @composed function make_leavepgroups_case(n_groups = Data.Integers(2, 8))
    groups = Int[]
    for g = 1:n_groups
      group_size = Data.produce!(Data.Integers(1, 6))
      append!(groups, fill(g, group_size))
    end
    p = Data.produce!(Data.Integers(1, n_groups - 1))
    return (groups, p)
  end

@testset "LeavePGroupsOut properties" begin
  @check max_examples = 150 rng = Xoshiro(14) function leavepgroupsout_enumerates_group_combinations(
    case = leavepgroups_case_gen,
  )
    groups, p = case
    N = length(groups)
    n_groups = length(unique(groups))
    X = reshape(collect(1:N), 1, N)

    cvs = partition(X, LeavePGroupsOut(p); groups = groups)

    test_group_sets = Set(Set(unique(groups[testindices(fold)])) for fold in cvs)

    length(cvs) == binomial(n_groups, p) &&
      length(test_group_sets) == binomial(n_groups, p) &&
      all(fold -> is_full_partition(fold, N), cvs) &&
      all(fold -> no_group_leakage(fold, groups), cvs) &&
      all(fold -> length(unique(groups[testindices(fold)])) == p, cvs)
  end
end

const predefined_split_case_gen =
  @composed function make_predefined_case(n_folds = Data.Integers(2, 5))
    N = Data.produce!(Data.Integers(n_folds, 30))
    # First n_folds observations are assigned one-per-fold to ensure all folds are present.
    # The remaining observations get random fold assignments.
    extra = [Data.produce!(Data.Integers(0, n_folds - 1)) for _ = (n_folds+1):N]
    test_fold = vcat(collect(0:(n_folds-1)), extra)
    return (test_fold, n_folds)
  end

@testset "PredefinedSplit properties" begin
  @check max_examples = 200 rng = Xoshiro(70) function predefined_split_valid(
    case = predefined_split_case_gen,
  )
    test_fold, n_folds = case
    N = length(test_fold)
    X = reshape(collect(1:N), 1, N)

    cvs = partition(X, PredefinedSplit(test_fold))

    length(folds(cvs)) == n_folds &&
      all(fold -> is_full_partition(fold, N), folds(cvs)) &&
      every_observation_tests_once(cvs, N)
  end
end
