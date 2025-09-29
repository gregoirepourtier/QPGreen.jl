using Test
using QPGreen, Bessels
using Printf
using StaticArrays
using LinearAlgebra

function QPGreenFunction_grad(params, points, idx, tol; verbose=false)

    vals_expansions = eigfunc_expansion_grad.(points, Ref(params); nb_terms=50_000_000)

    grid_sizes = [2^i for i ∈ idx]
    for grid_size ∈ grid_sizes
        value, grad, cache = init_qp_green_fft(params, grid_size; grad=true)

        if verbose
            println("Grid size: ", grid_size)
        end
        for i ∈ eachindex(points)
            grad_G = grad_qp_green(points[i], params, grad, cache; nb_terms=1_000_000)
            res_eig_x1, res_eig_x2 = vals_expansions[i]
            if verbose
                str_err_x1 = @sprintf "%.2E" abs(grad_G[1] - res_eig_x1)/abs(res_eig_x1)
                str_err_x2 = @sprintf "%.2E" abs(grad_G[2] - res_eig_x2)/abs(res_eig_x2)
                println("Point P", i, " and res: ", grad_G[1], " and error: ", str_err_x1)
                println("Point P", i, " and res: ", grad_G[2], " and error: ", str_err_x2)
                println(" ")
            end
            @test abs(res_eig_x1 - grad_G[1]) / abs(res_eig_x1) < tol
            @test abs(res_eig_x2 - grad_G[2]) / abs(res_eig_x2) < tol
        end
        if verbose
            println(" ")
        end
    end
end

function QPGreenFunction_grad_smooth(params, points, idx, tol; verbose=false)

    vals_expansions = eigfunc_expansion_grad.(points, Ref(params); nb_terms=50_000_000)

    for i ∈ eachindex(points)
        singularity = im / 4 * params.k * Bessels.hankelh1(1, params.k * norm(points[i])) / norm(points[i])
        vals_expansions[i] += singularity .* points[i]
    end


    grid_sizes = [2^i for i ∈ idx]
    for grid_size ∈ grid_sizes
        value, grad, cache = init_qp_green_fft(params, grid_size; grad=true)

        if verbose
            println("Grid size: ", grid_size)
        end
        for i ∈ eachindex(points)
            grad_G0 = grad_smooth_qp_green(points[i], params, grad, cache; nb_terms=1_000_000)
            res_eig_x1, res_eig_x2 = vals_expansions[i]
            if verbose
                str_err_x1 = @sprintf "%.2E" abs(grad_G0[1] - res_eig_x1)/abs(res_eig_x1)
                str_err_x2 = @sprintf "%.2E" abs(grad_G0[2] - res_eig_x2)/abs(res_eig_x2)
                println("Point P", i, " and res: ", grad_G0[1], " and error: ", str_err_x1)
                println("Point P", i, " and res: ", grad_G0[2], " and error: ", str_err_x2)
                println(" ")
            end
            @test abs(res_eig_x1 - grad_G0[1]) / abs(res_eig_x1) < tol
            @test abs(res_eig_x2 - grad_G0[2]) / abs(res_eig_x2) < tol
        end
        if verbose
            println(" ")
        end
    end
end

@testset "Gradient of the QP Green Function" begin

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
    QPGreenFunction_grad(params, points, 6:10, 1e-2; verbose=verbose)
    QPGreenFunction_grad_smooth(params, points, 7:10, 1e-3; verbose=verbose)

    verbose ? println("========= Tests 2 ==========") : nothing
    params = (alpha=0.3, k=5, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_grad(params, points, 6:10, 1e-2; verbose=verbose)
    QPGreenFunction_grad_smooth(params, points, 7:10, 1e-3; verbose=verbose)

    verbose ? println("========= Tests 3 ==========") : nothing
    params = (alpha=√2, k=50, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_grad(params, points, 7:10, 1e-2; verbose=verbose)
    QPGreenFunction_grad_smooth(params, points, 7:10, 5e-2; verbose=verbose)

    verbose ? println("========= Tests 4 ==========") : nothing
    params = (alpha=-√2, k=100, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_grad(params, points, 8:10, 5e-2; verbose=verbose)
    QPGreenFunction_grad_smooth(params, points, 8:10, 5e-2; verbose=verbose)
end
