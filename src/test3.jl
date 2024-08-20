include("GreenFunction.jl")
using FFTW, LinearAlgebra

## Another Test Fourier basis ##
N = 2
alpha, c, c̃, k, ε = (0.3, 0.6, 1.0, 10.0, 0.1)

grid_X, grid_Y = GreenFunction.gen_grid_FFT(π, c̃, N)
total_pts = (2 * N)^2
set_of_pt_grid = GreenFunction.get_grid_pts(grid_X, grid_Y, total_pts)

f(x) = 2 * sin(x[1] + x[2])
eval_f = zeros(size(grid_X))
cpt = 1
for i ∈ 1:(2 * N)
    for j ∈ 1:(2 * N)
        eval_f[i, j] = f(set_of_pt_grid[cpt, :])
        cpt += 1
    end
end
eval_f

fft_res = 1 / (2 * √(π * c̃)) .* fft(eval_f, 1)

res_fft1 = zeros(Complex{Float64}, size(eval_f))
res_tmp = 0
for j ∈ 1:(2 * N)
    for component ∈ 1:(2 * N)
        for i ∈ 1:(2 * N)
            res_tmp += eval_f[i, j] * 1 / (2 * √(π * c̃)) *
                       exp(-im * (component - 1) * (i - 1) * π / N - im * (j - 1) * π * (i - 1) / N) # 1 / (2 * √(π * c̃)) * exp(-im * 2 * π * (i - 1) * (component - 1) / (2 * N))# eval_f[i, j] * (1 / (2 * √(π * c̃)) * exp(-im * i * (component - 1) + im * j * (j - 1) * π / c̃)) # -GreenFunction.fourier_basis([component - 1, j - 1], i, j, c̃)
        end
        res_fft1[component, j] = res_tmp
        res_tmp = 0
    end
end

res_fft1
fft_res

norm(res_fft1) == norm(fft_res)



# fft_res = 1 / (2 * √(π * c̃)) .* fft(eval_f, 1)

# component1 = 2
# component2 = 1
# res_tmp = 0
# for i ∈ 1:(2 * N)
#     res_tmp += eval_f[i, component2] * 1 / (2 * √(π * c̃)) * exp(-im * 2 * π * (i - 1) * (component1 - 1) / (2 * N))
#     # exp(-2 * π * im * ((m - 1) * (component1 - 1) / size(A, 1) + (n - 1) * (component2 - 1) / size(A, 1)))
#     # exp(-im * (i - 1) * (component1 - 1) * π / N - im * (component2 - 1) * π * (component2 - 1) / N)
# end

# t1 = fft_res[component1, component2]
# t2 = res_tmp # fft_res

# norm(t1), norm(t2)
# isapprox(norm(t1), norm(t2); rtol=2e-1)

# # norm(res_fft1) == norm(fft_res)



fft_res = 1 / (2 * √(π * c̃)) .* fft(eval_f, 1)

res_fft1 = zeros(Complex{Float64}, size(eval_f))
res_tmp = 0
for j ∈ 1:(2 * N)
    for component ∈ 1:(2 * N)
        for i ∈ 1:(2 * N)
            res_tmp += eval_f[i, j] * 1 / (2 * √(π * c̃)) *
                       exp(-im * (component - 1) * (i - 1) * π / N - im * (j - 1) * π * (i - 1) / N) # 1 / (2 * √(π * c̃)) * exp(-im * 2 * π * (i - 1) * (component - 1) / (2 * N))# eval_f[i, j] * (1 / (2 * √(π * c̃)) * exp(-im * i * (component - 1) + im * j * (j - 1) * π / c̃)) # -GreenFunction.fourier_basis([component - 1, j - 1], i, j, c̃)
        end
        res_fft1[component, j] = res_tmp
        res_tmp = 0
    end
end

res_fft1
fft_res

norm(res_fft1) == norm(fft_res)

res_fft_vec = zeros(Complex{Float64}, 2 * N)
res_tmp = 0
column = 2
for component ∈ 1:(2 * N)
    for i ∈ 1:(2 * N)
        res_tmp += eval_f[i, column] * 1 / (2 * √(π * c̃)) *
                   exp(-im * (i - 1) * (component - 1) * π / N - im * (column - 1) * π * (column - 1) / N)
    end
    res_fft_vec[component] = res_tmp
    res_tmp = 0
end
res_fft_vec
