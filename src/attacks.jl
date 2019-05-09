function FGSM(x, ∇x, ϵ)
    sgn∇x = sign.(∇x)
    x̂ = x + ϵ * sgn∇x
    x̂ = min.(1, x̂)
    x̂ = max.(0, x̂)
    return x̂
end
