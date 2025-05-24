using Test
using QPGreen
using Printf
using StaticArrays

function QPGreenFunction_der(params, points, idx, tol; verbose=false)

    vals_expansions = eigfunc_expansion_derivative.(points, Ref(params); nb_terms=50_000_000)

    grid_sizes = [2^i for i ∈ idx]
    for grid_size ∈ grid_sizes
        value, grad, cache = init_qp_green_fft(params, grid_size; derivative=true)

        if verbose
            println("Grid size: ", grid_size)
        end
        for i ∈ eachindex(points)
            grad_G = grad_qp_green(points[i], params, grad, cache; nb_terms=1_000_000)
            res_eig_x1, res_eig_x2 = vals_expansions[i]
            @test abs(res_eig_x1 - grad_G[1]) < tol
            @test abs(res_eig_x2 - grad_G[2]) < tol
            if verbose
                str_err_x1 = @sprintf "%.2E" abs(G_x1 - res_eig_x1)/abs(res_eig_x1)
                str_err_x2 = @sprintf "%.2E" abs(G_x2 - res_eig_x2)/abs(res_eig_x2)
                println("Point P", i, " and res: ", G_x1, " and error: ", str_err_x1)
                println("Point P", i, " and res: ", G_x2, " and error: ", str_err_x2)
                println(" ")
            end
        end
        if verbose
            println(" ")
        end
    end
end

function QPGreenFunction_der_smooth(params, points, idx, tol; verbose=false)

    vals_expansions = image_expansion_derivative_smooth.(points, Ref(params); nb_terms=50_000_000)

    grid_sizes = [2^i for i ∈ idx]
    for grid_size ∈ grid_sizes
        value, grad, cache = init_qp_green_fft(params, grid_size; derivative=true)

        if verbose
            println("Grid size: ", grid_size)
        end
        for i ∈ eachindex(points)
            grad_G0 = grad_smooth_qp_green(points[i], params, grad, cache; nb_terms=1_000_000)
            res_eig_x1, res_eig_x2 = vals_expansions[i]
            @test abs(res_eig_x1 - grad_G0[1]) < tol
            @test abs(res_eig_x2 - grad_G0[2]) < tol
            if verbose
                str_err_x1 = @sprintf "%.2E" abs(G_x1 - res_eig_x1)/abs(res_eig_x1)
                str_err_x2 = @sprintf "%.2E" abs(G_x2 - res_eig_x2)/abs(res_eig_x2)
                println("Point P", i, " and res: ", G_x1, " and error: ", str_err_x1)
                println("Point P", i, " and res: ", G_x2, " and error: ", str_err_x2)
                println(" ")
            end
        end
        if verbose
            println(" ")
        end
    end
end

@testset "Derivatives of the QP Green Function" begin

    P1 = SVector(0.01π, 0.001)
    P2 = SVector(0.01π, 0.01)
    P3 = SVector(0.01π, 0.1)
    P4 = SVector(0.5π, 0.001)
    P5 = SVector(0.5π, 0.01)
    P6 = SVector(0.5π, 0.1)
    points = [P1, P2, P3, P4, P5, P6]

    verbose = false

    verbose ? println("========= Tests 1 ==========") : nothing
    params = (alpha=0.3, k=√10, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_der(params, points, 5:10, 5e-2; verbose=verbose)
    QPGreenFunction_der_smooth(params, points, 5:10, 5e-2; verbose=verbose)

    verbose ? println("========= Tests 2 ==========") : nothing
    params = (alpha=0.3, k=5, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_der(params, points, 5:10, 5e-2; verbose=verbose)
    QPGreenFunction_der_smooth(params, points, 5:10, 5e-2; verbose=verbose)

    verbose ? println("========= Tests 3 ==========") : nothing
    params = (alpha=√2, k=50, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_der(params, points, 7:10, 5e-1; verbose=verbose)
    QPGreenFunction_der_smooth(params, points, 7:10, 5e-1; verbose=verbose)

    verbose ? println("========= Tests 4 ==========") : nothing
    params = (alpha=-√2, k=100, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_der(params, points, 8:10, 2e-1; verbose=verbose)
    QPGreenFunction_der_smooth(params, points, 8:10, 2e-1; verbose=verbose)
end
