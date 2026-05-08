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
