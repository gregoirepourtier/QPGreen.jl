using QPGreen
using Test

function expansion_eval_test(z, csts::NamedTuple, tol)
    G_im = image_expansion(z, csts; period=2π, nb_terms=15_000_000)
    G_eigfun = eigfunc_expansion(z, csts; period=2π, nb_terms=500_000)

    @test isapprox(G_im, G_eigfun, atol=tol)
end

@testset "Test suite expansions (eval)" begin
    z = [1.0, 2.0]
    csts = (k=10.0, alpha=0.3)

    expansion_eval_test(z, csts, 1e-5)
    # Add more tests here
end

function expansion_grad_eval_test(z, csts::NamedTuple, tol)
    G_im_x1, G_im_x2 = image_expansion_grad(z, csts; period=2π, nb_terms=10_000_000)
    G_eigfun_x1, G_eigfun_x2 = eigfunc_expansion_grad(z, csts; period=2π, nb_terms=500_000)

    @test isapprox(G_im_x1, G_eigfun_x1, atol=tol)
    @test isapprox(G_im_x2, G_eigfun_x2, atol=tol)
end

@testset "Test suite expansions (gradient)" begin
    z = [1.0, 2.0]
    csts = (k=10.0, alpha=0.3)

    expansion_grad_eval_test(z, csts, 1e-4)
    # Add more tests here
end

function expansion_hess_eval_test(z, csts::NamedTuple, tol)
    G_im_x1x1, G_im_x1x2, G_im_x2x2 = image_expansion_hess(z, csts; period=2π, nb_terms=10_000_000)
    G_eigfun_x1x1, G_eigfun_x1x2, G_eigfun_x2x2 = eigfunc_expansion_hess(z, csts; period=2π, nb_terms=500_000)

    @test isapprox(G_im_x1x1, G_eigfun_x1x1, atol=tol)
    @test isapprox(G_im_x1x2, G_eigfun_x1x2, atol=tol)
    @test isapprox(G_im_x2x2, G_eigfun_x2x2, atol=tol)
end

@testset "Test suite expansions (hessian)" begin
    z = [1.0, 2.0]
    csts = (k=10.0, alpha=0.3)

    expansion_hess_eval_test(z, csts, 1e-3)
    # Add more tests here
end
