using QPGreen
using Test

function expansion_eval_test(z, csts::NamedTuple, tol)
    G_im = image_expansion(z, csts; period=2π, nb_terms=150_000_000)
    G_eigfun = eigfunc_expansion(z, csts; period=2π, nb_terms=500_000)

    @test isapprox(G_im, G_eigfun, atol=tol)
end

@testset "Test suite expansions (eval)" begin
    z = [1.0, 2.0]
    csts = (k=10.0, α=0.3)

    expansion_eval_test(z, csts, 1e-6)
    # Add more tests here
end

function expansion_derivative_eval_test(z, csts::NamedTuple, tol)
    G_im_x1, G_im_x2 = image_expansion_derivative(z, csts; period=2π, nb_terms=10_000_000)
    G_eigfun_x1, G_eigfun_x2 = eigfunc_expansion_derivative(z, csts; period=2π, nb_terms=500_000)

    @test isapprox(G_im_x1, G_eigfun_x1, atol=tol)
    @test isapprox(G_im_x2, G_eigfun_x2, atol=tol)
end

@testset "Test suite expansions (gradient)" begin
    z = [1.0, 2.0]
    csts = (k=10.0, α=0.3)

    expansion_derivative_eval_test(z, csts, 1e-4)
    # Add more tests here
end
