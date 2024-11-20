using QPGreen
using Test

function expansion_test(z, k, α, tol)
    G_im = image_expansion(z, k, α; period=2π, nb_terms=300_000_000)
    G_eigfun = eigfunc_expansion(z, k, α; period=2π, nb_terms=1_000_000)

    @test isapprox(G_im, G_eigfun, atol=tol)
end

@testset "Test suite expansions" begin
    z = [1.0, 2.0]
    k, α = (10.0, 0.3)

    expansion_test(z, k, α, 1e-6)
    # Add more tests here
end
