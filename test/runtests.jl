using Knet
using AutoGrad
import AutoGrad: gcheck
using Taarruz
using Test
using .Iterators


@testset "mnist" begin
    include(Knet.dir("data", "mnist.jl"))
    global dtrn, dtst = mnistdata(; xtype=Taarruz._atype)
    @test dtrn.length == 60000
    @test dtst.length == 10000
    @test dtrn.xsize[1:end-1] == dtst.xsize[1:end-1] == (28, 28, 1)
    @test length(dtrn) == div(dtrn.length, dtrn.batchsize)
    @test length(dtst) == div(dtst.length, dtst.batchsize)
end


@testset "lenet" begin
    global lenet = Lenet()
    x, y = first(dtst); x = param(x)
    @test gcheck(lenet, x, y; atol=0.05)
    @test typeof(lenet) == Taarruz.Chain
    @test size(lenet(x)) == (10, dtst.batchsize)
    @test accuracy(lenet, dtst) < 0.20
    progress!(adam(lenet, dtrn))
    global tstacc = accuracy(lenet, dtst)
    @test tstacc > 0.95
end


@testset "fgsm" begin
    x, y = first(dtst)

    ϵ = 0.2
    x̂s = FGSM(lenet, ϵ, x, y)
    x̂ = x̂s[1]
    @test length(x̂s) == 1
    @test size(x̂) == size(x)
    @test typeof(x̂) == typeof(x)
    @test maximum(x̂) == 1
    @test minimum(x̂) == 0

    example(x,y,ϵ=0.2,f=lenet) = FGSM(f,ϵ,x,y)[1]
    abuse(x,y,ϵ=0.2,f=lenet; o...) = accuracy(f(example(x,y)), y; o...)
    abuse(d::Knet.Data, ϵ=0.2, f=lenet) = sum(abuse(x,y,ϵ,f; average=false) for (x,y) in d) / d.length

    fgsmacc = abuse(dtst)
    @test tstacc - fgsmacc > 0.2

    fgsmacc = map(ϵi->abuse(x,y,ϵi), 0.1:0.05:0.5)
    @test issorted(fgsmacc, rev=true)

    ϵ = 0.2
    xp = param(x)
    J = @diff lenet(xp, y)
    ∇x = grad(J, xp)
    x̂ = min.(1, max.(0, x + ϵ * sign.(∇x)))
    @test length(Taarruz.∇xJ(lenet, x, y)) == 1
    @test ∇x ≈ Taarruz.∇xJ(lenet, x, y)[1]
    @test x̂ ≈ FGSM(x, ∇x, ϵ)
    @test x̂ ≈ FGSM(lenet, ϵ, x, y)[1]
end
