include("GreenFunction.jl")

Yε(x) = cos(x)
Yε_der(x) = -sin(x)
Yε_der_2nd(x) = -cos(x)

x = [3.0, 1.0]
GreenFunction.Φ₁(x, Yε_der, Yε_der_2nd)
GreenFunction.Φ₂(x, Yε_der, Yε_der_2nd)

grid_size = 5

Dc_x1 = (-π, π)
Dc_x2 = (-1.0, 1.0)

total_pts = 4 * grid_size^2

grid_X, grid_Y = GreenFunction.gen_grid_FFT(Dc_x1[2], Dc_x2[2], grid_size)

set_of_pt_grid = zeros(total_pts, 2);
set_of_pt_grid[:, 1] .= 1
set_of_pt_grid[:, 1] .* ones(100) *2
@time for i ∈ 1:total_pts
    set_of_pt_grid[i, 1] = grid_X[i]
    set_of_pt_grid[i, 2] = grid_Y[i]
end

res = zeros(total_pts)
for i ∈ 1:total_pts
    @views res[i] = norm(set_of_pt_grid[i,:])

    GreenFunction.Φ₁.(set_of_pt_grid, Yε_der, Yε_der_2nd)
end
res
norm(set_of_pt_grid[2,:])



# 1. Preparation step
evaluation_Φ₁ = reshape(GreenFunction.Φ₁.(set_of_pt_grid, Yε_der, Yε_der_2nd), (2 * grid_size, 2 * grid_size))


# evaluation_Φ₂ = reshape(Φ₂.(set_of_pt_grid, Yε_der, Yε_der_2nd), (2 * grid_size, 2 * grid_size))











A = reshape(collect(1.0:16), (4, 4))

# A = collect(1.0:16);
component = 5;

res1 = fft(A, 1) # [component]
ifft(res1, 1)


res = 0
for i ∈ 1:eachindex(A)
    res += exp(-im * 2 * π * (i - 1) * (component - 1) / length(A)) * A[i]
end
res

using LinearAlgebra

grid_X, grid_Y = GreenFunction.gen_grid_FFT(1.0, 1.0, 5) ; 

@time res = zeros(10,10);
function test!(res, grid_X, grid_Y)
    for i ∈ 1:10
        for j ∈ 1:10
            res[i,j] = norm((grid_X[i,j], grid_Y[i,j]))
        end
    end
end

@time test!(res, grid_X, grid_Y)

res

norm([grid_X[5,7], grid_Y[5,7]])