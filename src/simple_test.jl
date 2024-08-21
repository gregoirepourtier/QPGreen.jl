# Simple test to check the correctness of the 2D FFT implementation

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
# eval_f = fftshift(eval_f)

### 2D FFT ###
fft_res_2D_API = 1 / (2 * √(π * c̃)) .* fft(eval_f)


res_fft_2D = zeros(Complex{Float64}, size(eval_f));
for component1 ∈ (-N):(N - 1), component2 ∈ (-N):(N - 1)
    res_tmp = 0
    for i ∈ (-N):(N - 1)
        for j ∈ (-N):(N - 1)
            res_tmp += eval_f[i + N + 1, j + N + 1] * 1 / (2 * √(π * c̃)) *
                       exp(-im * i * component1 * π / N - im * j * π * component2 / N)
            if component1 == -1 && component2 == -2
                println("i: ", i, " j: ", j, " res_tmp: ", res_tmp)
            end
        end
    end
    res_fft_2D[component1 + N + 1, component2 + N + 1] = res_tmp
end

t1 = res_fft_2D
t2 = fftshift(fft_res_2D_API)
isapprox(norm(t1), norm(t2); rtol=1e-10)


t2
fftshift(t2)
fftshift(t1)



## Normal Indices
res_fft_2D = zeros(Complex{Float64}, size(eval_f))
for component1 ∈ 1:(2 * N), component2 ∈ 1:(2 * N)
    res_tmp = 0
    for i ∈ 1:1:(2 * N)
        for j ∈ 1:(2 * N)
            res_tmp += eval_f[i, j] * 1 / (2 * √(π * c̃)) *
                       exp(-im * (i - 1) * (component1 - 1) * π / N - im * (j - 1) * π * (component2 - 1) / N)
        end
    end
    res_fft_2D[component1, component2] = res_tmp
end
res_fft_2D
