using Pkg
Pkg.activate("test/Project.toml")
using Test, GreenFunction, LinearAlgebra, GLMakie, SpecialFunctions, FFTW

begin
    χ_der(x) = GreenFunction.build_χ_der(x, c̃, c)
    Yε(x) = GreenFunction.build_Yε(x, ε)
    Yε_der(x) = GreenFunction.build_Yε_der(x, ε)
    Yε_der_2nd(x) = GreenFunction.build_Yε_der_2nd(x, ε)
end;

X, Y = (0.0, 0.01 * 2π);
r, θ = (√(X^2 + Y^2), atan(Y, X));

α, c, c̃, k, ε = (√2 / (2 * π), 0.6, 1.0, 2 / (2 * π), 0.5)
csts = (α, c, c̃, k, ε)

# Test to match parameter from the paper (Linton, 1998)
(X / (2π) == 0.0, Y / (2π) == 0.01, k * (2π) == 2, α * (2π) == √2, r < (2π))


params = (α, c, c̃, k)
GreenFunction.fm_method_preparation(params, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=32);
# GreenFunction.fm_method_calculation(x, params, preparation_result, Yε; α=alpha, k=k, nb_terms=32)

res_eig = GreenFunction.green_function_eigfct_exp((X, Y); k=k, α=α, nb_terms=1000)
res_img = GreenFunction.green_function_img_exp((X, Y); k=k, α=α, nb_terms=200000)


# Tests
grid_size = 2
total_pts = (2 * grid_size)^2 # (2N)² points

# Generate the grid
grid_X, grid_Y = GreenFunction.gen_grid_FFT(0.5, 0.5, grid_size);
set_of_pt_grid = GreenFunction.get_grid_pts(grid_X, grid_Y, total_pts)
N, M = size(grid_X)
@assert N==M==2*grid_size "Problem dimensions"

#### 1. Preparation step ####
_evaluation_Φ₁ = GreenFunction.Φ₁(set_of_pt_grid, Yε_der, Yε_der_2nd, total_pts)
evaluation_Φ₁ = transpose(reshape(_evaluation_Φ₁, (N, M)))

res1 = 1 / (2 * √(π * c̃)) .* fftshift(fft(evaluation_Φ₁))
(2 * √(π * c̃)) .* ifft(fftshift(res1))

# _evaluation_Φ₂ = view(set_of_pt_grid, :, 1) .* _evaluation_Φ₁
# evaluation_Φ₂ = transpose(reshape(_evaluation_Φ₂, (N, M)))

# GreenFunction.Φ₁(set_of_pt_grid[1, :], Yε_der, Yε_der_2nd, total_pts)
x_test = set_of_pt_grid[3, :]
norm(x_test)

norm(set_of_pt_grid)

local_phi_1(x) = (2 + log(x)) * Yε_der(x) / x + Yε_der_2nd(x) * log(x)

1 / (2 * √(π * c̃)) .* fftshift(fft(reshape(local_phi_1.(norm.(ls_pt)), (N, M))))


ls_pt = []
for i ∈ 1:size(set_of_pt_grid, 1)
    push!(ls_pt, set_of_pt_grid[i, :])
end

# Φ̂₁ⱼ = 1 / (2 * √(π * c̃)) .* fft(evaluation_Φ₁)
# Φ̂₂ⱼ = 1 / (2 * √(π * c̃)) .* fft(evaluation_Φ₂)
# Φ̂₁ⱼ = fftshift(Φ̂₁ⱼ)
# Φ̂₂ⱼ = fftshift(Φ̂₂ⱼ)
