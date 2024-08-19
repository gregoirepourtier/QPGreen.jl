include("GreenFunction.jl")
using FFTW
using LinearAlgebra

## Test functions ϕ₁, ϕ₂
Yε(x) = cos(x)
Yε_der(x) = -sin(x)
Yε_der_2nd(x) = -cos(x)

x = [3.0 1.0]
GreenFunction.Φ₁(x, Yε_der, Yε_der_2nd, 1)[1] == 1.171657934917535
GreenFunction.Φ₂(x, Yε_der, Yε_der_2nd, 1)[1] == 3.5149738047526045

## Test FFT and IFFT
A = reshape(collect(1.0:16.0), (4, 4))
# A = collect(1.0:16);

component = 1;

res1 = fft(A)

# res = 0
# for i ∈ 1:length(A)
#     res += exp(-im * 2 * π * (i - 1) * (component - 1) / length(A)) * A[i]
# end
# res

j = 1
component = 4
res = 0
for i ∈ 1:size(A, 1)
    res += exp(-im * 2 * π * (i - 1) * (component - 1) / size(A, 1)) * A[i, j]
end
res



res = zeros(Complex{Float64}, size(A))
res_tmp = 0
for component1 ∈ 1:size(A, 1)
    for component2 ∈ 1:size(A, 1)
        for m ∈ 1:size(A, 1)
            for n ∈ 1:size(A, 1)
                res_tmp += A[m, n] *
                           exp(-2 * π * im * ((m - 1) * (component1 - 1) / size(A, 1) + (n - 1) * (component2 - 1) / size(A, 1)))
            end
        end
        res[component1, component2] = res_tmp
        res_tmp = 0
    end
end
res
res1


for j ∈ 1:size(A, 1)
    for component ∈ 1:size(A, 1)
        for i ∈ 1:size(A, 1)
            res_tmp += exp(-im * 2 * π * ((i - 1) * (component - 1) / size(A, 1))) * A[i, j]
        end
        res[component, j] = res_tmp
        res_tmp = 0
    end
end


## Test Fourier basis
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

res = zeros(Complex{Float64}, size(eval_f))
res_tmp = 0
for j ∈ (-N):(N - 1)
    for component ∈ (-N):(N - 1)
        for i ∈ (-N):(N - 1)
            res_tmp += eval_f[i + N + 1, j + N + 1] * exp(-im * 2 * π * ((i - 1) * (component - 1) / (2 * N)))
        end
        res[component + N + 1, j + N + 1] = res_tmp
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
