using Dates
using DataSplits
using Test

@testset "TimeSplit" begin
  s = TimeSplitOldest()

  # All same date — the entire dataset must go to one cohort
  d2 = fill(Date(2020, 1, 1), 10)
  result = partition(d2, s; train = 60, test = 40)
  train, test = result.train, result.test
  @test (length(train) == 10 && isempty(test)) || (isempty(train) && length(test) == 10)

  # Two date groups — group boundary respected
  d3 = vcat(fill(Date(2020, 1, 1), 5), fill(Date(2020, 1, 2), 5))
  result = partition(d3, s; train = 60, test = 40)
  train, test = result.train, result.test
  @test (length(train) == 5 || length(train) == 10)
  @test length(train) + length(test) == 10
end

@testset "TimeSplitNewest edge cases" begin
  s = TimeSplitNewest()

  # All same date — the entire dataset must go to one cohort
  d_same = fill(Date(2020, 1, 1), 10)
  result = partition(d_same, s; train = 60, test = 40)
  train, test = result.train, result.test
  @test (length(train) == 10 && isempty(test)) || (isempty(train) && length(test) == 10)

  # Two date groups — group boundary respected; newest date must land in train
  d_two = vcat(fill(Date(2020, 1, 1), 5), fill(Date(2020, 1, 2), 5))
  result = partition(d_two, s; train = 60, test = 40)
  train, test = result.train, result.test
  @test (length(train) == 5 || length(train) == 10)
  @test length(train) + length(test) == 10
  if length(train) == 5
    @test all(i -> i in 6:10, train)   # obs 6-10 carry the newer date
  end
end
