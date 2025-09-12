using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen
using Printf, LinearAlgebra
using BenchmarkTools: @btime

Z = (0.002π, 0.0);
params = (alpha=0.3, k=1.0, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8);

grid_size = 1024;
val_interp, cache = init_qp_green_fft(params, grid_size);
eval_qp_green(Z, params, val_interp, cache; nb_terms=32);

res_eig = 0.7685069487 + 0.1952423542im
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    val_interp, cache = init_qp_green_fft(params, i)
    res_fm = eval_qp_green(Z, params, val_interp, cache; nb_terms=32)
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end

### Modified part
grid_size = 32;
val_interp, cache = init_qp_green_fft_mod(params, grid_size);
eval_qp_green_mod(Z, params, val_interp, cache; nb_terms=32);

res_eig = 0.7685069487 + 0.1952423542im
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    val_interp, cache = init_qp_green_fft_mod(params, i)
    res_fm = eval_qp_green_mod(Z, params, val_interp, cache; nb_terms=32)
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end


@btime eval_qp_green($Z, $params, $val_interp, $cache; nb_terms=32);
@btime eval_qp_green_mod($Z, $params, $val_interp, $cache; nb_terms=32);