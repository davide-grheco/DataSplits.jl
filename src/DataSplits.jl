module DataSplits


include("core.jl")
include("strategies/random.jl")
include("strategies/LazyKennardStone.jl")
include("strategies/KennardStone.jl")

export split
export LazyCADEXSplit, LazyKennardStoneSplit, KennardStoneSplit
export RandomSplit

end
