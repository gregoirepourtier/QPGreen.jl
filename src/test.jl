using FastGaussQuadrature, LinearAlgebra

x, w = gausslegendre(3)

f(x) = x^4

I = dot(w, f.(x))
I = w' * f.(x)

int(x) = x^5 / 5

I ≈ int(1) - int(-1)


a = 5
b = 10

c_of_var(x, a, b) = (b - a) / 2 * f((b - a) / 2 * x + (a + b) / 2)

I2 = w' * c_of_var.(x, a, b)

I2 ≈ int(10) - int(5)



for i ∈ 1:3, j ∈ 1:2
    println(i, " ", j)
end


using LazyGrids
x, y = (1, 1)
N = 11

grid_test = ndgrid(0:(1 / (N - 1)):x, 0:(1 / (N - 1)):y)

grid_test[1]
grid_test[2]

size(grid_test[1])

for i ∈ 1:length(grid_test[1])
    println(grid_test[1][i], " ", grid_test[2][i])
    if i % N == 0
        println()
    end
end





## FFT

using FFTW, LinearAlgebra

Yε(x) = cos(x)
Yε_der(x) = -sin(x)
Yε_der_2nd(x) = -cos(x)


function Φ₁(x, Yε_der, Yε_der_2nd)
    x_norm = norm(x)

    return (2 + log(x_norm)) * Yε_der(x_norm) / x_norm + Yε_der_2nd(x_norm) * log(x_norm)
end
Φ₂(x, Yε_der, Yε_der_2nd) = x[1] * Φ₁(x, Yε_der, Yε_der_2nd)


x = [3.0, 1.0]
Φ₁(x, Yε_der, Yε_der_2nd)
Φ₂(x, Yε_der, Yε_der_2nd)


x, y = (1, 1)
N = 11
using LazyGrids
grid_test = ndgrid(0:(1 / (N - 1)):x, 0:(1 / (N - 1)):y)
grid_test[1]
grid_test[2]
# getnnodes(grid)
# set_of_pt_grid = Vector{Float64}[]
# for i ∈ 1:length(grid_test[1])
#     push!(set_of_pt_grid, [grid_test[1][i], grid_test[2][i]])
# end
(grid_test[1], grid_test[2])

set_of_pt_grid

Φ₁.(set_of_pt_grid,Yε_der, Yε_der_2nd)

Φ₁.(set_of_pt_grid,Yε_der, Yε_der_2nd)
# [grid_test[1], grid_test[2]]



using FFTW

A = reshape(collect(1.0:16),(4,4))
# A = collect(1.0:16);

# component = 5
res1 = fft(A, 1)#[component]

ifft(res1, 1)

for i in 1:16
    println(res1[i])
end

res = 0
for i ∈ 1:length(A)
    res += exp(-im * 2 * π * (i - 1) * (component - 1) / length(A)) * A[i]
end
res


using LazyGrids
x, y = (1, 1)
N = 11

grid_test = ndgrid(0:(1 / (N - 1)):x, 0:(1 / (N - 1)):y)
grid_test[1]
grid_test[2]

function test(x)
    x₁, x₂ = x

    x₁ .+ x₂
end

@time test(grid_test)