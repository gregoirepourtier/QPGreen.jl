using Test
using QPGreen
using Printf

function QPGreenFunction_eval(params, points, idx, tol; verbose=false)

    vals_expansions = eigfunc_expansion.(points, Ref(params); nb_terms=50_000_000)

    grid_sizes = [2^i for i ∈ idx]
    for grid_size ∈ grid_sizes
        interp, cache = fm_method_preparation(params, grid_size)

        if verbose
            println("Grid size: ", grid_size)
        end
        for i ∈ eachindex(points)
            G_x = fm_method_calculation(points[i], params, interp, cache; nb_terms=1_000_000)
            res_eig = vals_expansions[i]
            @test abs(res_eig - G_x) < tol
            if verbose
                str_err = @sprintf "%.2E" abs(G_x - res_eig)/abs(res_eig)
                println("Point P", i, " and res: ", G_x, " and error: ", str_err)
            end
        end
        if verbose
            println(" ")
        end
    end
end

@testset "Paper FFT-based algorithm, B. Zhang and R. Zhang" begin

    P1 = (0.01π, 0.0)
    P2 = (0.01π, 0.01)
    P3 = (0.5π, 0.0)
    P4 = (0.5π, 0.01)
    points = [P1, P2, P3, P4]

    verbose = false

    # Refer to Table 1
    verbose ? println("========= Table 1 ==========") : nothing
    params = (α=0.3, k=√10, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, points, 5:10, 1e-3; verbose=verbose)

    # Refer to Table 2
    verbose ? println("========= Table 2 ==========") : nothing
    params = (α=0.3, k=5, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, points, 5:10, 1e-3; verbose=verbose)

    # Refer to Table 3
    verbose ? println("========= Table 3 ==========") : nothing
    params = (α=√2, k=50, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, points, 7:10, 1e-2; verbose=verbose)

    # Refer to Table 4
    verbose ? println("========= Table 4 ==========") : nothing
    params = (α=-√2, k=100, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, points, 8:10, 1e-2; verbose=verbose)
end

@testset "Paper Green's function for the 2D Helmholtz equation in periodic domains, C.M Linton" begin
    P1 = (0.0, 0.01 * 2π)
    point = [P1]

    verbose = false

    # Refer to Table 2
    verbose ? println("========= Table 2 ==========") : nothing
    β, k, d = (√2 / 2π, 1 / π, 2π)

    # Test to match parameter from the paper (Linton, 1998)
    @test (P1[1] / d == 0.0, P1[2] / d == 0.01, k * d == 2, β * d == √2) == (true, true, true, true)

    params = (α=β, k=k, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, point, 5:10, 1e-3; verbose=verbose)

    # Refer to Table 3
    verbose ? println("========= Table 3 ==========") : nothing
    β, k, d = (5√2 / 2π, 5 / π, 2π)

    # Test to match parameter from the paper (Linton, 1998)
    @test (P1[1] / d == 0.0, P1[2] / d == 0.01, k * d == 10, β * d == 5√2) == (true, true, true, true)

    params = (α=β, k=k, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, point, 5:10, 1e-3; verbose=verbose)

    # Refer to Table 4
    verbose ? println("========= Table 4 ==========") : nothing
    β, k, d = (3 / 2π, 1 / π, 2π)

    # Test to match parameter from the paper (Linton, 1998)
    @test (P1[1] / d == 0.0, P1[2] / d == 0.01, k * d == 2, β * d == 3) == (true, true, true, true)

    params = (α=β, k=k, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, point, 5:10, 1e-3; verbose=verbose)

    # Refer to Table 5
    verbose ? println("========= Table 5 ==========") : nothing
    β, k, d = (√2 / 2π, 1 / π, 2π)

    P2 = (0.1 * 2π, π)
    P3 = (0.5 * 2π, π)
    points = [P2, P3]

    # Test to match parameter from the paper (Linton, 1998)
    @test (P2[1] / d == 0.1, P2[2] / d == 0.5, k * d == 2, β * d == √2) == (true, true, true, true)
    @test (P3[1] / d == 0.5, P3[2] / d == 0.5, k * d == 2, β * d == √2) == (true, true, true, true)

    params = (α=β, k=k, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, points, 5:10, 1e-10; verbose=verbose) # here since abs(z[2]) > c, we can use the eigenfunction expansion directly
end
