module Taarruz

using Knet
_etype = gpu() >= 0 ? Float32 : Float64
_atype = gpu() >= 0 ? KnetArray{_etype} : Array{_etype}


include("attacks.jl"); export FGSM
include("lenet.jl")

end # module
