function ∇xJ(f, args...)
    args = collect(args)
    args = map(a->isa(a, FloatArray) ? param(a) : a, args)
    J = @diff f(args...)
    ∇ = map(arg->grad(J, arg), args)
    ∇ = filter(i->isa(i, FloatArray), ∇)
end


function FGSM(x::FloatArray, ∇x::FloatArray, ϵ::T) where T <: AbstractFloat
    sgn∇x = sign.(∇x)
    x̂ = x + ϵ * sgn∇x
    x̂ = min.(1, x̂)
    x̂ = max.(0, x̂)
    return x̂
end


function FGSM(f, ϵ::T, args...) where T <: AbstractFloat
    args = collect(args)
    ∇x  = ∇xJ(f, args...)
    x = filter(i->isa(i, FloatArray), args)
    x̂ = [FGSM(xi, ∇xi, ϵ) for (xi, ∇xi) in zip(x, ∇x)]
end
