include("GreenFunction.jl")
using FFTW, LinearAlgebra

N = 2
alpha, c, c̃, k, ε = (0.3, 0.6, 1.0, 10.0, 0.1)

grid_X, grid_Y = GreenFunction.gen_grid_FFT(π, c̃, N)
total_pts = (2 * N)^2
set_of_pt_grid = GreenFunction.get_grid_pts(grid_X, grid_Y, total_pts)

f(x) = 2 * sin(x[1] + x[2])

eval_f = zeros(size(grid_X))
cpt = 1
for i ∈ 1:(2 * N), j ∈ 1:(2 * N)
    eval_f[i, j] = f(set_of_pt_grid[cpt, :])
    cpt += 1
end
eval_f

# 1D FFT
fft_res = 1 / (2 * √(π * c̃)) .* fft(eval_f)

res_tmp = 0
component1 = 1
component2 = 3

for i ∈ 1:(2 * N)
    for j ∈ 1:(2 * N)
        res_tmp += eval_f[i, j] * 1 / (2 * √(π * c̃)) *
                   exp(-im * (i - 1) * (component1 - 1) * π / N - im * (j - 1) * π * (component2 - 1) / N)
    end
end

t1 = res_tmp
t2 = fft_res[component1, component2]
isapprox(norm(t1), norm(t2); rtol=1e-10)

# 2D FFT
fft_res_2D = 1 / (2 * √(π * c̃)) .* fft(eval_f, 1)

# res_fft_vec = zeros(Complex{Float64}, 2 * N)
res_tmp = 0
component1 = 1
component2 = j = 3

for i ∈ 1:(2 * N)
    res_tmp += eval_f[i, j] * 1 / (2 * √(π * c̃)) *
               exp(-im * (i - 1) * (component1 - 1) * π / N - im * (j - 1) * π * (component2 - 1) / N)
end

t1 = res_tmp
t2 = fft_res[component1, component2]
isapprox(norm(t1), norm(t2); rtol=1e-10)


