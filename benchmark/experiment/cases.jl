using DataSplits

struct Case
  name::String
  family::Symbol
  eager_pair::Union{String,Nothing}
  run::Function
end

Case(name, family, run::Function) = Case(name, family, nothing, run)

const TT = (train = 0.8, test = 0.2)

const CASES = Case[
  Case(
    "KennardStoneSplit",
    :distance_eager,
    d -> partition(d.X, KennardStoneSplit(); TT...),
  ),
  Case("SPXYSplit", :distance_eager, d -> partition(d.X, SPXYSplit(); target = d.y, TT...)),
  Case("OptiSimSplit", :distance_eager, d -> partition(d.X, OptiSimSplit(); TT...)),
  Case(
    "LazyKennardStoneSplit",
    :distance_lazy,
    "KennardStoneSplit",
    d -> partition(d.X, LazyKennardStoneSplit(); TT...),
  ),
  Case(
    "LazySPXYSplit",
    :distance_lazy,
    "SPXYSplit",
    d -> partition(d.X, LazySPXYSplit(); target = d.y, TT...),
  ),
  Case(
    "LazyOptiSimSplit",
    :distance_lazy,
    "OptiSimSplit",
    d -> partition(d.X, LazyOptiSimSplit(); TT...),
  ),
  Case("sphere_exclusion", :clustering, d -> sphere_exclusion(d.X; radius = 0.3)),
  Case("RandomSplit", :simple, d -> partition(d.X, RandomSplit(); TT...)),
  Case(
    "TargetPropertyHigh",
    :simple,
    d -> partition(d.X, TargetPropertyHigh(); target = d.y, TT...),
  ),
  Case(
    "TargetPropertyLow",
    :simple,
    d -> partition(d.X, TargetPropertyLow(); target = d.y, TT...),
  ),
  Case("TimeSplitOldest", :simple, d -> partition(d.X, TimeSplit(); time = d.times, TT...)),
]

const CASE_BY_NAME = Dict(c.name => c for c in CASES)
