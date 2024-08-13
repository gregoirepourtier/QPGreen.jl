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



for i in 1:3, j in 1:2
    println(i, " ", j)
end


using LazyGrids
x, y = (1, 1)
N = 11

grid_test = ndgrid(0:(1 / (N - 1)):x, 0:(1 / (N - 1)):y)

grid_test[1]
grid_test[2]

size(grid_test[1])

for i in 1:length(grid_test[1])
    println(grid_test[1][i], " ", grid_test[2][i])
    if i%N == 0
        println()
    end
end