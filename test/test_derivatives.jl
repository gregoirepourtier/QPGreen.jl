# using QPGreen
# using Test

# function expansion_test(z, k, α)
#     G_prime_eigfun = QPGreen.analytical_derivative(z, k, α; period=2π, nb_terms=100_000)

# end

# z = [1.0, 0.0]
# k, α = (10.0, 0.3)

# expansion_test(z, k, α)



# @testset "Test suite expansions" begin
#     z = [1.0, 2.0]
#     k, α = (10.0, 0.3)

#     expansion_test(z, k, α)
#     # Add more tests here
# end

using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen
using Printf, LinearAlgebra

Z = (0.002π, 0.0);
params = (α=0.3, k=1.0, c=0.6, c̃=1.0, ε=0.4341, order=8);

grid_size = 1024;
preparation_result, interp, cache = fm_method_preparation_derivative(params, grid_size);
fm_method_calculation_derivative(Z, params, preparation_result, interp, cache; nb_terms=32)

res_eig = 0.7685069487 + 0.1952423542im
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    # preparation_result, interp, cache = QPGreen.fm_method_preparation(params, i)
    # res_fm = QPGreen.fm_method_calculation(Z, params, preparation_result, interp, cache; nb_terms=32)
    preparation_result, interp, cache = fm_method_preparation_derivative(params, i)
    res_fm = fm_method_calculation_derivative(Z, params, preparation_result, interp, cache; nb_terms=32)[1]
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end
