include("GreenFunction.jl")

Yε(x) = cos(x)
Yε_der(x) = -sin(x)
Yε_der_2nd(x) = -cos(x)

x = [3.0 1.0]
GreenFunction.Φ₁(x, Yε_der, Yε_der_2nd, 1)
GreenFunction.Φ₂(x, Yε_der, Yε_der_2nd, 1)


using FFTW
A = reshape(collect(1.0:16.0), (4, 4))
# A = collect(1.0:16);

component = 2;

res1 = fft(A, 1) # [component]
# ifft(res1, 1)

res = 0
for i ∈ 1:length(A)
    res += exp(-im * 2 * π * (i - 1) * (component - 1) / length(A)) * A[i]
end
res

j = 1
component = 4
res = 0
for i ∈ 1:size(A, 1)
    res += exp(-im * 2 * π * (i - 1) * (component - 1) / size(A, 1)) * A[i, j]
end
res



res = zeros(Complex{Float64}, size(A))
res_tmp = 0
for j ∈ 1:size(A, 1)
    for component ∈ 1:size(A, 1)
        for i ∈ 1:size(A, 1)
            res_tmp += exp(-im * 2 * π * ((i - 1) * (component - 1) / size(A, 1))) * A[i, j]
        end
        res[component, j] = res_tmp
        res_tmp = 0
    end
end
res


using Interpolations
f(x, y) = im * log(x + y) + x
xs = 1:0.2:5
ys = 2:0.1:5
A = [f(x, y) for x ∈ xs, y ∈ ys]

# linear interpolation
interp_linear = linear_interpolation((xs, ys), A)
interp_linear(3, 2) == f(3, 2) # exactly log(3 + 2)
isapprox(interp_linear(3.1, 2.1), f(3.1, 2.1); rtol=1e-3) # approximately log(3.1 + 2.1)

# cubic spline interpolation
interp_cubic = cubic_spline_interpolation((xs, ys), A)
interp_cubic(3, 2) # exactly log(3 + 2)
isapprox(interp_cubic(3.1, 2.1), f(3.1, 2.1); rtol=1e-7) # approximately log(3.1 + 2.1)


using FastGaussQuadrature, LinearAlgebra

x, w = gausslegendre(5)
f(x) = x^4 * (1 - x)^4
quad(f::T, x, a, b) where {T} = (b - a) / 2 * f((b - a) / 2 * x + (a + b) / 2)
I = dot(w, quad.(f, x, 0, 1))
I ≈ 1/630
