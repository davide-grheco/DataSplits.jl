# ------------------------------------------------------------------
# Generators
# ------------------------------------------------------------------

const kfold_case_gen = @composed function make_kfold_case(N = Data.Integers(2, 200))
  k = Data.produce!(Data.Integers(2, N))
  return (N, k)
end

const shuffle_split_case_gen = @composed function make_shuffle_split_case(
  N = Data.Integers(2, 100),
  n_splits = Data.Integers(1, 10),
)
  n_train = Data.produce!(Data.Integers(1, N - 1))
  return (N, n_splits, n_train, N - n_train)
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

const stratified_case_gen =
  @composed function make_stratified_case(n_classes = Data.Integers(2, 8))
    k = Data.produce!(Data.Integers(2, 10))
    labels = Symbol[]
    for c = 1:n_classes
      extra = Data.produce!(Data.Integers(0, 20))
      append!(labels, fill(Symbol(:c, c), k + extra))
    end
    return (labels, k)
  end

const leavepout_case_gen = @composed function make_leavepout_case(N = Data.Integers(2, 10))
  p = Data.produce!(Data.Integers(1, N - 1))
  return (N, p)
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

const predefined_split_case_gen =
  @composed function make_predefined_case(n_folds = Data.Integers(2, 5))
    N = Data.produce!(Data.Integers(n_folds, 30))
    extra = [Data.produce!(Data.Integers(0, n_folds - 1)) for _ = (n_folds+1):N]
    test_fold = vcat(collect(0:(n_folds-1)), extra)
    return (test_fold, n_folds)
  end

const repeated_case_gen = @composed function make_repeated_case(N = Data.Integers(4, 100))
  k = Data.produce!(Data.Integers(2, min(N, 10)))
  n_repeats = Data.produce!(Data.Integers(2, 5))
  return (N, k, n_repeats)
end

# Stratified variant uses 2 classes with N÷2 members each — requires k ≤ N÷2.
const repeated_strat_case_gen =
  @composed function make_repeated_strat_case(N = Data.Integers(4, 100))
    k = Data.produce!(Data.Integers(2, max(2, N ÷ 2)))
    n_repeats = Data.produce!(Data.Integers(2, 5))
    return (N, k, n_repeats)
  end

const stratified_grouped_case_gen =
  @composed function make_stratified_grouped_case(n_groups = Data.Integers(2, 20))
    n_classes = Data.produce!(Data.Integers(2, 4))
    k = Data.produce!(Data.Integers(2, n_groups))
    groups = Int[]
    labels = Symbol[]
    for g = 1:n_groups
      group_size = Data.produce!(Data.Integers(n_classes, n_classes * 3))
      class_assign = [Symbol(:c, mod1(i, n_classes)) for i = 1:group_size]
      append!(groups, fill(g, group_size))
      append!(labels, class_assign)
    end
    return (groups, labels, k)
  end

const grouped_shuffle_cv_case_gen =
  @composed function make_grouped_shuffle_cv_case(n_groups = Data.Integers(2, 20))
    n_splits = Data.produce!(Data.Integers(2, 8))
    groups = Int[]
    for g = 1:n_groups
      group_size = Data.produce!(Data.Integers(1, 10))
      append!(groups, fill(g, group_size))
    end
    return (groups, n_splits)
  end

# ------------------------------------------------------------------
# Plain KFold
# ------------------------------------------------------------------

@testset "KFold properties" begin
  @check max_examples = 300 rng = Xoshiro(10) function kfold_is_valid_cv(
    case = kfold_case_gen,
  )
    N, k = case
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, KFold(k))
    length(cvs) == k &&
      is_full_partition(cvs, N) &&
      every_observation_tests_once(cvs, N) &&
      fold_test_sizes_balanced(cvs)
  end
end

# ------------------------------------------------------------------
# Stratified KFold
# ------------------------------------------------------------------

@testset "StratifiedKFold properties" begin
  @check max_examples = 300 rng = Xoshiro(12) function stratified_kfold_balances_classes(
    case = stratified_case_gen,
  )
    labels, k = case
    N = length(labels)
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, StratifiedKFold(k); target = labels)
    length(cvs) == k &&
      is_full_partition(cvs, N) &&
      every_observation_tests_once(cvs, N) &&
      class_counts_balanced_across_test_folds(cvs, labels)
  end
end

# ------------------------------------------------------------------
# Group-aware KFold family
# Both require groups=, no group leakage, every group tests once.
# StratifiedGroupKFold additionally requires target= for class balance.
# ------------------------------------------------------------------

@testset "GroupKFold properties" begin
  @check max_examples = 300 rng = Xoshiro(11) function groupkfold_is_valid(
    case = grouped_case_gen,
  )
    groups, k = case
    N = length(groups)
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, GroupKFold(k); groups = groups)
    length(cvs) == k &&
      is_full_partition(cvs, N) &&
      all(fold -> no_group_leakage(fold, groups), cvs) &&
      every_observation_tests_once(cvs, N) &&
      every_group_tests_once(cvs, groups)
  end
end

@testset "StratifiedGroupKFold properties" begin
  @check max_examples = 200 rng = Xoshiro(73) function stratified_group_kfold_is_valid(
    case = stratified_grouped_case_gen,
  )
    groups, labels, k = case
    N = length(groups)
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, StratifiedGroupKFold(k); groups = groups, target = labels)
    length(cvs) == k &&
      is_full_partition(cvs, N) &&
      all(fold -> no_group_leakage(fold, groups), cvs) &&
      every_observation_tests_once(cvs, N) &&
      every_group_tests_once(cvs, groups)
  end
end

# ------------------------------------------------------------------
# Shuffle resample family
# ShuffleSplit and StratifiedShuffleSplit share: n_splits folds,
# is_full_partition, has_correct_split_size.
# GroupShuffleSplitCV adds no_group_leakage (size may overshoot).
# ------------------------------------------------------------------

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
      is_full_partition(cvs, N) &&
      has_correct_split_size(cvs, n_train, n_test)
  end
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
      is_full_partition(cvs, N) &&
      has_correct_split_size(cvs, n_train, n_test)
  end
end

@testset "GroupShuffleSplitCV properties" begin
  @check max_examples = 200 rng = Xoshiro(74) function group_shuffle_cv_no_leakage(
    case = grouped_shuffle_cv_case_gen,
  )
    groups, n_splits = case
    N = length(groups)
    X = reshape(collect(1:N), 1, N)
    cvs = partition(
      X,
      GroupShuffleSplitCV(n_splits);
      groups = groups,
      train = 1,
      test = N - 1,
      rng = Xoshiro(42),
    )
    length(folds(cvs)) == n_splits &&
      is_full_partition(cvs, N) &&
      all(fold -> no_group_leakage(fold, groups), cvs)
  end
end

# ------------------------------------------------------------------
# Repeated KFold family
# Both produce k * n_repeats folds; each repeat block covers 1:N once.
# ------------------------------------------------------------------

@testset "RepeatedKFold properties" begin
  @check max_examples = 200 rng = Xoshiro(75) function repeated_kfold_is_valid(
    case = repeated_case_gen,
  )
    N, k, n_repeats = case
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, RepeatedKFold(k; n_repeats = n_repeats); rng = Xoshiro(42))
    fs = folds(cvs)
    length(fs) == k * n_repeats && all(
      r -> begin
        slice = fs[(r*k+1):((r+1)*k)]
        sort(reduce(vcat, [testindices(f) for f in slice])) == 1:N
      end,
      0:(n_repeats-1),
    )
  end
end

@testset "RepeatedStratifiedKFold properties" begin
  @check max_examples = 200 rng = Xoshiro(76) function repeated_strat_kfold_is_valid(
    case = repeated_strat_case_gen,
  )
    N, k, n_repeats = case
    labels = [Symbol(:c, mod1(i, 2)) for i = 1:N]
    X = reshape(collect(1:N), 1, N)
    cvs = partition(
      X,
      RepeatedStratifiedKFold(k; n_repeats = n_repeats);
      target = labels,
      rng = Xoshiro(42),
    )
    fs = folds(cvs)
    length(fs) == k * n_repeats && all(
      r -> begin
        slice = fs[(r*k+1):((r+1)*k)]
        sort(reduce(vcat, [testindices(f) for f in slice])) == 1:N
      end,
      0:(n_repeats-1),
    )
  end
end

# ------------------------------------------------------------------
# BootstrapSplit — unique: train indices have duplicates (sampling with replacement)
# ------------------------------------------------------------------

@testset "BootstrapSplit properties" begin
  @check max_examples = 200 rng = Xoshiro(77) function bootstrap_oob_covers_remainder(
    N = Data.Integers(5, 80),
    n_splits = Data.Integers(1, 20),
  )
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, BootstrapSplit(n_splits); rng = Xoshiro(42))
    length(folds(cvs)) == n_splits && all(folds(cvs)) do f
      train = trainindices(f)
      test = testindices(f)
      length(train) == N &&
        allunique(test) &&
        isempty(intersect(Set(train), Set(test))) &&
        sort(collect(union(Set(train), Set(test)))) == 1:N
    end
  end
end

# ------------------------------------------------------------------
# LeavePOut / LeavePGroupsOut
# ------------------------------------------------------------------

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
      is_full_partition(cvs, N)
  end
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
      is_full_partition(cvs, N) &&
      all(fold -> no_group_leakage(fold, groups), cvs) &&
      all(fold -> length(unique(groups[testindices(fold)])) == p, cvs)
  end
end

# ------------------------------------------------------------------
# PredefinedSplit
# ------------------------------------------------------------------

@testset "PredefinedSplit properties" begin
  @check max_examples = 200 rng = Xoshiro(70) function predefined_split_valid(
    case = predefined_split_case_gen,
  )
    test_fold, n_folds = case
    N = length(test_fold)
    X = reshape(collect(1:N), 1, N)
    cvs = partition(X, PredefinedSplit(test_fold))
    length(folds(cvs)) == n_folds &&
      is_full_partition(cvs, N) &&
      every_observation_tests_once(cvs, N)
  end
end

# ------------------------------------------------------------------
# CombinatorialPurgedKFold
# ------------------------------------------------------------------

@testset "CombinatorialPurgedKFold properties" begin
  @check max_examples = 100 rng = Xoshiro(91) function cpcv_valid_cv(case = kfold_case_gen)
    N, k = case
    k = min(k, 6)
    times = collect(1:N)
    X = reshape(times, 1, N)
    cvs = partition(X, CombinatorialPurgedKFold(k, 1); time = times)
    is_full_partition(cvs, N)
  end
end
