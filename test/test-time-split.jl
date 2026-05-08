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
