using Random
using Supposition
using Supposition.Data

using DataSplits: trainindices, valindices, testindices, TrainTestSplit, TrainValTestSplit

# ---------------------------------------------------------------------
# Common helpers — usable from any test file
# ---------------------------------------------------------------------


function has_valid_index_set(indices, N)
  return all(i -> 1 <= i <= N, indices) && allunique(indices)
end

"""
    is_full_partition(result, N)

PBT wrapper — uses `trainindices`/`testindices` accessors.
Used in `@check` blocks in property test files.
"""
function is_full_partition(result::TrainTestSplit, N)
  train = trainindices(result)
  test = testindices(result)

  return has_valid_index_set(train, N) &&
         has_valid_index_set(test, N) &&
         isempty(intersect(train, test)) &&
         sort(vcat(train, test)) == collect(1:N)
end

function is_full_partition(result::TrainValTestSplit, N)
  train = trainindices(result)
  val = valindices(result)
  test = testindices(result)

  return has_valid_index_set(train, N) &&
         has_valid_index_set(val, N) &&
         has_valid_index_set(test, N) &&
         isempty(intersect(train, val)) &&
         isempty(intersect(train, test)) &&
         isempty(intersect(val, test)) &&
         sort(vcat(train, val, test)) == collect(1:N)
end

function is_full_partition(cvs::CrossValidationSplit, N)
  return all(fold -> is_full_partition(fold, N), folds(cvs))
end

function cv_has_expected_number_of_folds(cvs::CrossValidationSplit, expected)
  return length(folds(cvs)) == expected
end

"""
  Check that partitions of a dataset complement each other and cover the full dataset.
"""
function cohorts_are_complements(result::TrainTestSplit, N)
  train = trainindices(result)
  test = testindices(result)

  return sort(test) == setdiff(1:N, train) && sort(train) == setdiff(1:N, test)
end

function cohorts_are_complements(result::TrainValTestSplit, N)
  train = trainindices(result)
  val = valindices(result)
  test = testindices(result)

  return sort(train) == setdiff(1:N, vcat(val, test)) &&
         sort(val) == setdiff(1:N, vcat(train, test)) &&
         sort(test) == setdiff(1:N, vcat(train, val))
end

"""
    is_disjoint(result)

Check that train and test indices do not overlap.
"""
function is_disjoint(result::TrainTestSplit)
  return isempty(intersect(trainindices(result), testindices(result)))
end

"""
    is_disjoint(result)

Check that train and test indices do not overlap.
"""
function is_disjoint(result::TrainValTestSplit)
  return isempty(intersect(trainindices(result), testindices(result))) &&
         isempty(intersect(valindices(result), testindices(result))) &&
         isempty(intersect(valindices(result), trainindices(result)))
end

"""
    is_disjoint(result)

Fallback for unnamed types or NamedTuples with `.train`/`.test` fields.
"""
function is_disjoint(result)
  return isempty(intersect(result.train, result.test))
end

function has_correct_split_size(result, n_train, n_test)
  return length(trainindices(result)) == n_train && length(testindices(result)) == n_test
end

function has_correct_split_size(result, n_train, n_val, n_test)
  return length(trainindices(result)) == n_train &&
         length(valindices(result)) == n_val &&
         length(testindices(result)) == n_test
end

function has_correct_split_size(cvs::CrossValidationSplit, n_train, n_test)
  return all(fold -> has_correct_split_size(fold, n_train, n_test), folds(cvs))
end

function has_correct_split_size(cvs::CrossValidationSplit, n_train, n_val, n_test)
  return all(fold -> has_correct_split_size(fold, n_train, n_val, n_test), folds(cvs))
end

"""
    total_size(result)

Return total number of samples in the split (train + test + val).
"""
total_size(result::TrainTestSplit) =
  length(trainindices(result)) + length(testindices(result))
total_size(result::TrainValTestSplit) =
  length(trainindices(result)) + length(valindices(result)) + length(testindices(result))

function no_group_leakage(result::TrainTestSplit, groups)
  train_groups = Set(groups[trainindices(result)])
  test_groups = Set(groups[testindices(result)])
  return isempty(intersect(train_groups, test_groups))
end

function no_group_leakage(result::TrainValTestSplit, groups)
  train_groups = Set(groups[trainindices(result)])
  val_groups = Set(groups[valindices(result)])
  test_groups = Set(groups[testindices(result)])
  return isempty(intersect(train_groups, val_groups)) &&
         isempty(intersect(train_groups, test_groups)) &&
         isempty(intersect(val_groups, test_groups))
end

function same_indices(a::TrainTestSplit, b::TrainTestSplit)
  return trainindices(a) == trainindices(b) && testindices(a) == testindices(b)
end

function same_indices(a::TrainValTestSplit, b::TrainValTestSplit)
  return trainindices(a) == trainindices(b) &&
         valindices(a) == valindices(b) &&
         testindices(a) == testindices(b)
end


# ---------------------------------------------------------------------
# PBT-specific helpers — require Supposition.jl generators
# ---------------------------------------------------------------------


function every_observation_tests_once(cvs, N)
  test_counts = zeros(Int, N)
  for fold in cvs
    for i in testindices(fold)
      test_counts[i] += 1
    end
  end
  all(==(1), test_counts)
end

function fold_test_sizes_balanced(cvs)
  sizes = [length(testindices(fold)) for fold in cvs]
  maximum(sizes) - minimum(sizes) <= 1
end

function every_group_tests_once(cvs, groups)
  group_counts = Dict(g => 0 for g in unique(groups))
  for fold in cvs
    for g in unique(groups[testindices(fold)])
      group_counts[g] += 1
    end
  end
  all(==(1), values(group_counts))
end

function class_counts_balanced_across_test_folds(cvs, labels)
  for c in unique(labels)
    counts = [count(==(c), labels[testindices(fold)]) for fold in cvs]
    maximum(counts) - minimum(counts) <= 1 || return false
  end
  return true
end

function no_time_value_split(result, times)
  train_times = Set(times[trainindices(result)])
  test_times = Set(times[testindices(result)])
  return isempty(intersect(train_times, test_times))
end

function oldest_train_before_test(result, times)
  train = trainindices(result)
  test = testindices(result)
  return all(times[i] <= times[j] for i in train, j in test)
end

function newest_train_after_test(result, times)
  train = trainindices(result)
  test = testindices(result)
  return all(times[i] >= times[j] for i in train, j in test)
end
