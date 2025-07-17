module DataSplits


include("core.jl")
include("utils.jl")
include("strategies/random.jl")
include("strategies/LazyKennardStone.jl")
include("strategies/KennardStone.jl")
include("strategies/SPXY.jl")

export split
export LazyCADEXSplit, LazyKennardStoneSplit, KennardStoneSplit
export SPXYSplit
export RandomSplit

end
