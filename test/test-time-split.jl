using Dates
using DataSplits
using Test

@testset "TimeSplit" begin
  # Simple case: unique dates
  d = Date(2020, 1, 1):Day(1):Date(2020, 1, 10)
  d = collect(d)
  s = TimeSplitOldest(0.6)
  result = split(d, s)
  train, test = result.train, result.test
  @test all(d[i] <= d[j] for i in train, j in test)
  @test length(train) + length(test) == 10
  @test length(train) >= 6

  # All same date
  d2 = fill(Date(2020, 1, 1), 10)
  result = split(d2, s)
  train, test = result.train, result.test
  @test (length(train) == 10 && isempty(test)) || (isempty(train) && length(test) == 10)

  # Ties: two groups
  d3 = vcat(fill(Date(2020, 1, 1), 5), fill(Date(2020, 1, 2), 5))
  result = split(d3, s)
  train, test = result.train, result.test
  @test (length(train) == 5 || length(train) == 10)
  @test length(train) + length(test) == 10

  # Newest in train
  s2 = TimeSplitNewest(0.5)
  result = split(d, s2)
  train, test = result.train, result.test
  @test all(d[i] >= d[j] for i in train, j in test)
end
