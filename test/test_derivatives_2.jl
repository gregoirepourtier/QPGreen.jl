using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen
using Printf, LinearAlgebra

# Z = (0.002π, 0.0);

Z = (0.002π, 0.01);
# Z = (0.05π, 0.01);
params = (α=0.3, k=1.0, c=0.6, c̃=1.0, ε=0.4341, order=8);
# params = (α=0.3, k=(√(10)), c=0.6, c̃=1.0, ε=0.4341, order=8);

res_der = analytical_derivative(Z, params; nb_terms=100_000_000)

# res = (0.050308789349485 + 0.10989902020665852im, 0.0 + 0.0im); # nb_terms=100_000_000 for Z = (0.002π, 0.0)
res = res_der[1] # (-24.654079665241486 + 0.06215272762758034im, -3.9322899312645165 - 0.00013269041328632292im); # nb_terms=100_000_000 for Z = (0.002π, 0.001)

grid_size = 1024;
preparation_result, interp, cache = fm_method_preparation_derivative(params, grid_size);
fm_method_calculation_derivative(Z, params, preparation_result, interp, cache; nb_terms=32)

res_eig = res[1] # 0.7685069487 + 0.1952423542im
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    # preparation_result, interp, cache = QPGreen.fm_method_preparation(params, i)
    # res_fm = QPGreen.fm_method_calculation(Z, params, preparation_result, interp, cache; nb_terms=32)
    preparation_result, interp, cache = fm_method_preparation_derivative(params, i)
    res_fm = fm_method_calculation_derivative(Z, params, preparation_result, interp, cache; nb_terms=32)[1]
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end
