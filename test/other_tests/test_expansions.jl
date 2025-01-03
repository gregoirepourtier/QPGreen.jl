using QPGreen
using Test

function expansion_eval_test(z, csts::NamedTuple, tol)
    G_im = image_expansion(z, csts; period=2π, nb_terms=300_000_000)
    G_eigfun = eigfunc_expansion(z, csts; period=2π, nb_terms=1_000_000)

    @test isapprox(G_im, G_eigfun, atol=tol)
end

@testset "Test suite expansions (eval)" begin
    z = [1.0, 2.0]
    csts = (k=10.0, α=0.3)

    expansion_eval_test(z, csts, 1e-6)
    # Add more tests here
end

@testset "Test suite expansions (gradient)" begin
    z = [1.0, 2.0]
    csts = (k=10.0, α=0.3)

    res1, res2 = eigfunc_expansion_derivative(z, csts; period=2π, nb_terms=1_000_000)
    @test isapprox(res1, 0.3771056242536212 + 0.412259643399506im)
    @test isapprox(res2, 0.3313275794704831 - 0.2308702935192195im)
    # Add more tests here
end
