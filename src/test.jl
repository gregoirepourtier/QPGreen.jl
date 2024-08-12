using FastGaussQuadrature, LinearAlgebra

x, w = gausslegendre(3)

f(x) = x^4

I = dot(w, f.(x))

int(x) = x^5 / 5

I ≈ int(1) - int(-1)


a = 5
b = 10

c_of_var(x, a, b) = (b - a) / 2 * f((b - a) / 2 * x + (a + b) / 2)

I2 = dot(w, c_of_var.(x, a, b))

I2 ≈ int(10) - int(5)



for i in 1:3, j in 1:2
    println(i, " ", j)
end