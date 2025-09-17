using Pkg
Pkg.activate("test/Project.toml")
using Test
using QPGreen
using Printf, LinearAlgebra
using BenchmarkTools: @btime
using SpecialFunctions

Z = (0.002π, 0.0);
# params = (alpha=0.3, k=1.0, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8);
params = (alpha=0.3, k=√10, c=0.6, c_tilde=1.0, epsilon=0.4341, order=8)

grid_size = 32;
val_interp, cache = init_qp_green_fft(params, grid_size);
eval_qp_green(Z, params, val_interp, cache; nb_terms=32);

res_eig = 0.7685069487 + 0.1952423542im; # - im / 4 * besselh(0, params.k * norm(Z))
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    val_interp, cache = init_qp_green_fft(params, i)
    res_fm = eval_qp_green(Z, params, val_interp, cache; nb_terms=32)
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end

### Asymptotic part
grid_size = 32;
val_interp, cache = QPGreen.init_qp_green_fft_asymptotic(params, grid_size);
QPGreen.eval_qp_green_asymptotic(Z, params, val_interp, cache; nb_terms=32);

res_eig = 0.7685069487 + 0.1952423542im; # - im / 4 * besselh(0, params.k * norm(Z));
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    val_interp, cache = QPGreen.init_qp_green_fft_asymptotic(params, i)
    res_fm = QPGreen.eval_qp_green_asymptotic(Z, params, val_interp, cache; nb_terms=32)
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end

@btime eval_qp_green($Z, $params, $val_interp, $cache; nb_terms=32);
@btime QPGreen.eval_qp_green_asymptotic($Z, $params, $val_interp, $cache; nb_terms=32);
@btime eval_smooth_qp_green($Z, $params, $val_interp; nb_terms=32);
@btime QPGreen.eval_smooth_qp_green_asymptotic($Z, $params, $val_interp, $cache; nb_terms=32);

## Smooth part
P1 = (0.01π, 0.0)
P2 = (0.01π, 0.01)
P3 = (0.5π, 0.0)
P4 = (0.5π, 0.01)

res_eig = eigfunc_expansion(P1, params; nb_terms=150_000_000) - im / 4 * besselh(0, params.k * norm(P1))
res_img = QPGreen.image_expansion_smooth(P1, params; nb_terms=150_000_000)
val_interp, cache = QPGreen.init_qp_green_fft_asymptotic(params, 1024);
res_fm = QPGreen.eval_smooth_qp_green_asymptotic(Z, params, val_interp, cache; nb_terms=32)
val_interp, cache = QPGreen.init_qp_green_fft(params, 1024);
res_fm = QPGreen.eval_smooth_qp_green(Z, params, val_interp; nb_terms=32)

# res_eig = res_img
@time for i ∈ [32, 64, 128, 256, 512, 1024]
    val_interp, cache = QPGreen.init_qp_green_fft(params, i)
    res_fm = QPGreen.eval_smooth_qp_green(Z, params, val_interp; nb_terms=32)
    str_err = @sprintf "%.2E" abs(res_fm - res_eig)/abs(res_eig)
    println("Grid size: ", i, " and res: ", res_fm, " and error: ", str_err)
end

function image_expansion_smooth_test(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the Green function by basic image expansion
    G = zero(Complex{eltype(z)})
    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)
        G += im / 4 * exp(im * period * α * -n) * Bessels.hankelh1(0, k * r₋ₙ) +
             im / 4 * exp(im * period * α * n) * Bessels.hankelh1(0, k * rₙ)
    end

    G
end
vals_expansions = image_expansion_smooth_test(P1, params; nb_terms=50_000_000)
