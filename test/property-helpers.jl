using Random
using Supposition
using Supposition.Data

using DataSplits: trainindices, valindices, testindices

function pbt_has_train_test_sizes(result, n_train, n_test)
  return length(trainindices(result)) == n_train && length(testindices(result)) == n_test
end

function pbt_has_train_val_test_sizes(result, n_train, n_val, n_test)
  return length(trainindices(result)) == n_train &&
         length(valindices(result)) == n_val &&
         length(testindices(result)) == n_test
end

function pbt_is_full_train_test_partition(result, N)
  train = trainindices(result)
  test = testindices(result)

  return isempty(intersect(train, test)) && sort(vcat(train, test)) == collect(1:N)
end

function pbt_is_full_train_val_test_partition(result, N)
  train = trainindices(result)
  val = valindices(result)
  test = testindices(result)

  return isempty(intersect(train, val)) &&
         isempty(intersect(train, test)) &&
         isempty(intersect(val, test)) &&
         sort(vcat(train, val, test)) == collect(1:N)
end

function pbt_no_group_leakage(result, groups)
  train_groups = Set(groups[trainindices(result)])
  test_groups = Set(groups[testindices(result)])

  return isempty(intersect(train_groups, test_groups))
end

function pbt_no_group_leakage_train_val_test(result, groups)
  train_groups = Set(groups[trainindices(result)])
  val_groups = Set(groups[valindices(result)])
  test_groups = Set(groups[testindices(result)])

  return isempty(intersect(train_groups, val_groups)) &&
         isempty(intersect(train_groups, test_groups)) &&
         isempty(intersect(val_groups, test_groups))
end

function pbt_same_train_test_indices(a, b)
  return trainindices(a) == trainindices(b) && testindices(a) == testindices(b)
end

function pbt_same_train_val_test_indices(a, b)
  return trainindices(a) == trainindices(b) &&
         valindices(a) == valindices(b) &&
         testindices(a) == testindices(b)
end

function pbt_every_observation_tests_once(cvs, N)
  test_counts = zeros(Int, N)
  for fold in cvs
    for i in testindices(fold)
      test_counts[i] += 1
    end
  end
  all(==(1), test_counts)
end

function pbt_fold_test_sizes_balanced(cvs)
  sizes = [length(testindices(fold)) for fold in cvs]
  maximum(sizes) - minimum(sizes) <= 1
end

function pbt_every_group_tests_once(cvs, groups)
  group_counts = Dict(g => 0 for g in unique(groups))
  for fold in cvs
    for g in unique(groups[testindices(fold)])
      group_counts[g] += 1
    end
  end
  all(==(1), values(group_counts))
end

function pbt_class_counts_balanced_across_test_folds(cvs, labels)
  for c in unique(labels)
    counts = [count(==(c), labels[testindices(fold)]) for fold in cvs]
    maximum(counts) - minimum(counts) <= 1 || return false
  end
  return true
end

function pbt_no_time_value_split(result, times)
  train_times = Set(times[trainindices(result)])
  test_times = Set(times[testindices(result)])

  return isempty(intersect(train_times, test_times))
end

function pbt_oldest_train_before_test(result, times)
  train = trainindices(result)
  test = testindices(result)

  return all(times[i] <= times[j] for i in train, j in test)
end

function pbt_newest_train_after_test(result, times)
  train = trainindices(result)
  test = testindices(result)

  return all(times[i] >= times[j] for i in train, j in test)
end
