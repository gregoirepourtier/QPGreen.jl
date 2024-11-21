using Test
using QPGreen
using Printf

function QPGreenFunction_eval(params, points, idx, tol; verbose=false)
    P1, P2, P3, P4 = points

    vals_expansions = eigfunc_expansion.([P1, P2, P3, P4], Ref(params); nb_terms=100_000_000)

    grid_sizes = [2^i for i ∈ idx]
    for grid_size ∈ grid_sizes
        preparation_result, interp, cache = fm_method_preparation(params, grid_size)

        if verbose
            println("Grid size: ", grid_size)
        end
        for i ∈ 1:4
            G_x = fm_method_calculation([P1, P2, P3, P4][i], params, preparation_result, interp, cache; nb_terms=1_000_000)
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

    # Refer to Table 1
    params = (α=0.3, k=√10, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, points, 5:10, 1e-3; verbose=false)

    # Refer to Table 2
    params = (α=0.3, k=5, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, points, 5:10, 1e-3; verbose=false)

    # Refer to Table 3
    params = (α=√2, k=50, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, points, 7:10, 1e-2; verbose=false)

    # Refer to Table 4
    params = (α=-√2, k=100, c=0.6, c̃=1.0, ε=0.4341, order=8)
    QPGreenFunction_eval(params, points, 8:10, 1e-2; verbose=false)
end

@testset "Paper Green's function for the 2D Helmholtz equation in periodic domains, C.M Linton" begin
    P1 = (0.01π, 0.0)

end
