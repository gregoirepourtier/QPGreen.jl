using QPGreen
using Test

# Test image_expansion
function test_expansions()

    z = [1.0, 2.0]
    k = 10.0
    α = 0.3

    G_im = image_expansion(z, k, α; period=2π, nb_terms=10_000_000)
    G_eigfun = eigfunc_expansion(z, k, α; period=2π, nb_terms=10_000)

    @test isapprox(G_im, G_eigfun, atol=1e-5)
end

test_expansions()
