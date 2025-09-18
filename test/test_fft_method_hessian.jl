using Test
using QPGreen, Bessels
using Printf
using StaticArrays
using LinearAlgebra

using Pkg
Pkg.activate("test/Project.toml")

function QPGreenFunction_hess(params, points, idx, tol; verbose=false)

    vals_expansions = eigfunc_expansion_hess.(points, Ref(params); nb_terms=50_000_000)

    grid_sizes = [2^i for i ∈ idx]
    for grid_size ∈ grid_sizes
        value, grad, hess, cache = init_qp_green_fft(params, grid_size; grad=true, hess=true)

        if verbose
            println("Grid size: ", grid_size)
        end
        for i ∈ eachindex(points)
            hess_G = hess_qp_green(points[i], params, hess, cache; nb_terms=1_000_000)
            res_eig_x1, res_eig_x2 = vals_expansions[i]
            if verbose
                str_err_x1 = @sprintf "%.2E" abs(hess_G[1] - res_eig_x1)/abs(res_eig_x1)
                str_err_x2 = @sprintf "%.2E" abs(hess_G[2] - res_eig_x2)/abs(res_eig_x2)
                println("Point P", i, " and res: ", hess_G[1], " and error: ", str_err_x1)
                println("Point P", i, " and res: ", hess_G[2], " and error: ", str_err_x2)
                println(" ")
            end
            @test abs(res_eig_x1 - hess_G[1]) / abs(res_eig_x1) < tol
            @test abs(res_eig_x2 - hess_G[2]) / abs(res_eig_x2) < tol
        end
        if verbose
            println(" ")
        end
    end
end

function QPGreenFunction_hess_smooth(params, points, idx, tol; verbose=false)

    vals_expansions = eigfunc_expansion_hess.(points, Ref(params); nb_terms=50_000_000)

    k = params.k

    for i ∈ eachindex(points)
        # Precompute common terms
        r = norm(points[i])
        r2 = r^2
        r3 = r2 * r
        kr = k * r

        # Compute Hankel functions once
        h1 = Bessels.hankelh1(1, kr)
        h2 = Bessels.hankelh1(2, kr)

        # Precompute common factors
        common_factor1 = -im / 4 * k^2 * (h1 / kr - h2) / r2
        common_factor2 = -im / 4 * k * h1 / r3

        # Extract coordinates
        x, y = points[i][1], points[i][2]
        x2, y2 = x^2, y^2
        xy = x * y

        # Compute matrix elements
        m11 = common_factor1 * x2 + common_factor2 * (r2 - x2)
        m12 = common_factor1 * xy - common_factor2 * xy
        m22 = common_factor1 * y2 + common_factor2 * (r2 - y2)

        singularity = SVector(m11, m12, m22)
        vals_expansions[i] -= singularity
    end


    grid_sizes = [2^i for i ∈ idx]
    for grid_size ∈ grid_sizes
        value, grad, hess, cache = init_qp_green_fft(params, grid_size; grad=true, hess=true)

        if verbose
            println("Grid size: ", grid_size)
        end
        for i ∈ eachindex(points)
            hess_G0 = hess_smooth_qp_green(points[i], params, hess, cache; nb_terms=1_000_000)
            res_eig_x1x1, res_eig_x1x2, res_eig_x2x2 = vals_expansions[i]
            if verbose
                str_err_x1x1 = @sprintf "%.2E" abs(hess_G0[1] - res_eig_x1x1)/abs(res_eig_x1x1)
                str_err_x1x2 = @sprintf "%.2E" abs(hess_G0[2] - res_eig_x1x2)/abs(res_eig_x1x2)
                str_err_x2x2 = @sprintf "%.2E" abs(hess_G0[3] - res_eig_x2x2)/abs(res_eig_x2x2)
                println("Point P", i, " and res: ", hess_G0[1], " and error: ", str_err_x1x1)
                println("Point P", i, " and res: ", hess_G0[2], " and error: ", str_err_x1x2)
                println("Point P", i, " and res: ", hess_G0[3], " and error: ", str_err_x2x2)
                println(" ")
            end
            @test abs(res_eig_x1x1 - hess_G0[1]) / abs(res_eig_x1x1) < tol
            @test abs(res_eig_x1x2 - hess_G0[2]) / abs(res_eig_x1x2) < tol
            @test abs(res_eig_x2x2 - hess_G0[3]) / abs(res_eig_x2x2) < tol
        end
        if verbose
            println(" ")
        end
    end
end

@testset "Hessian of the QP Green Function" begin

    P1 = SVector(0.01π, 0.001)
    P2 = SVector(0.01π, 0.01)
    P3 = SVector(0.01π, 0.1)
    P4 = SVector(0.5π, 0.001)
    P5 = SVector(0.5π, 0.01)
    P6 = SVector(0.5π, 0.1)
    points = [P1, P2, P3, P4, P5, P6]

    verbose = true

    verbose ? println("========= Tests 1 ==========") : nothing
    params = (alpha=0.3, k=√10, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_hess(params, points, 6:10, 4e-2; verbose=verbose)
    QPGreenFunction_hess_smooth(params, points, 7:10, 2e-3; verbose=verbose)

    verbose ? println("========= Tests 2 ==========") : nothing
    params = (alpha=0.3, k=5, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_hess(params, points, 6:10, 2e-2; verbose=verbose)
    QPGreenFunction_hess_smooth(params, points, 7:10, 1e-3; verbose=verbose)

    verbose ? println("========= Tests 3 ==========") : nothing
    params = (alpha=√2, k=50, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_hess(params, points, 7:10, 1e-2; verbose=verbose)
    QPGreenFunction_hess_smooth(params, points, 7:10, 5e-2; verbose=verbose)

    verbose ? println("========= Tests 4 ==========") : nothing
    params = (alpha=-√2, k=100, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)
    QPGreenFunction_hess(params, points, 8:10, 3e-2; verbose=verbose)
    QPGreenFunction_hess_smooth(params, points, 8:10, 2e-2; verbose=verbose)
end
