include("GreenFunction.jl")
using FFTW, LinearAlgebra


## Test Fourier basis ##
N = 2
c̃ = 1.0

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


fft_res = fft(eval_f, 1)


# Computation Fourier coefficients
res = zeros(Complex{Float64}, size(eval_f))
res_tmp = 0
for j₂ ∈ (-N):(N - 1)
    for component ∈ 1:(2 * N)
        for j₁ ∈ (-N):(N - 1)
            res_tmp += eval_f[j₁ + N + 1, j₂ + N + 1] * exp(-im * 2 * π * ((j₁ - 1) * (component - 1) / (2 * N)))
        end
        res[component, j₂ + N + 1] = res_tmp
        res_tmp = 0
    end
end

fft_res
res

fftshift(fft_res)


norm(res) == norm(fft_res)


res = zeros(Complex{Float64}, size(A))
res_tmp = 0
for component1 ∈ 1:(2 * N)
    for component2 ∈ 1:(2 * N)
        for j₁ ∈ (-N):N
            for j₂ ∈ (-N):N
                res_tmp += f([j₁, j₂]) * -GreenFunction.fourier_basis([component1 - 1, component2 - 1], j₁, j₂, c̃)
            end
        end
        res[component1, component2] = res_tmp
        res_tmp = 0
    end
end

res


N = 5
A = collect(1:2*N)
res_fft = fft(A)

res = zeros(Complex{Float64}, 2*N)
for component ∈ 1:2*N
    for i ∈ 1:2*N
        res[component] += exp(-im * 2 * π * (i - 1) * (component - 1) / length(A)) * A[i]
    end
end

res
res_fft

norm(res) ≈ norm(res_fft)