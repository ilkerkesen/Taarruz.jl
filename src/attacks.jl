"""
    ∇xJ(f, args...)

Returns gradients of the loss function (f) w.r.t. your parameters (args).
"""
function ∇xJ(f, args...)
    args = collect(args)
    args = map(a->isa(a, FloatArray) ? param(a) : a, args)
    J = @diff f(args...)
    ∇ = map(arg->grad(J, arg), args)
    ∇ = filter(i->isa(i, FloatArray), ∇)
end


"""
This function implements FGSM (Fast Gradient Sign Method)
 (arxiv.org/abs/1412.6572). FGSM is a white-box attack which means the attacker
 has full access to the model  including its parameters. In this attack, we
 simply take advantage of the  gradients of the loss with respect to input data.
 We first apply sign function to the input gradients, scale it, and then add
 those gradients to the input  data to get the perturbed data which will
 maximize the loss value. See lenet-fgsm notebook for example usage.

    FGSM(x::FloatArray, ∇x::FloatArray, ϵ;
         minval=0.0, maxval=1.0, targeted=false)

Applies FGSM for given input array (x), the gradient of your model's loss w.r.t.
input array (∇x) and ϵ. minval and maxval stands for clipping. If you want to
perform a targeted attack, use targeted=true.

    FGSM(f, ϵ, args...; minval=0.0, maxval=1.0, targeted=false)

Applies FGSM for given loss function (f), ϵ and arguments for the loss function.
"""
function FGSM(x::FloatArray, ∇x::FloatArray, ϵ;
              minval=0.0, maxval=1.0, targeted=false)
    T = eltype(x)
    sgn∇x = sign.(∇x)
    x̂ = x + T(ϵ) * sgn∇x * (-1.)^targeted
    x̂ = min.(T(maxval), x̂)
    x̂ = max.(T(minval), x̂)
    return x̂
end


function FGSM(f, ϵ, args...; minval=0.0, maxval=1.0, targeted=false)
    args = collect(args)
    ∇x  = ∇xJ(f, args...)
    x = filter(i->isa(i, FloatArray), args)
    x̂ = [FGSM(xi, ∇xi, ϵ; minval=minval, maxval=maxval, targeted=targeted)
         for (xi, ∇xi) in zip(x, ∇x)]
end


"""
    IterativeFGSM(f, ϵ, T::Int, args...;
                  minval=0.0, maxval=1.0, targeted=false)

Applies Iterative FGSM attack. Simply collects and returns adversarial
attack for all timesteps. f is a closure function which calls your models
loss function and only accept parameters you want to get gradients.
"""
function IterativeFGSM(f, ϵ, T::Int, args...; minval=0.0, maxval=1.0)
    x̂ = collect(args)
    history = []
    for t = 1:T
        x̂ = FGSM(f, ϵ/T, x̂...; minval=minval, maxval=maxval)
        push!(history, x̂)
    end
    return history
end


function MomentumIterativeFGSM(f, ϵ, μ, T::Int, args...; minval=0.0, maxval=1.0)
    x̂ = collect(args)
    F = eltype(first(x̂))
    α = F(ϵ/T); μ = F(μ)
    history = []
    g = [F(0.0) for i = 1:length(x̂)]
    for t = 1:T
        ∇x̂ = ∇xJ(f, x̂...)
        g = [(μ*gi .+ ∇x̂i/norm(∇x̂i,1)) for (∇x̂i,gi) in zip(∇x̂, g)]
        x̂ = [FGSM(x̂i,gi,α; minval=minval, maxval=maxval)
             for (x̂i,gi) in zip(x̂, g)]
        push!(history, x̂)
    end
    return history
end
