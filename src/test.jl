include("GreenFunction.jl")
using FFTW

Yε(x) = cos(x)
Yε_der(x) = -sin(x)
Yε_der_2nd(x) = -cos(x)

x = [3.0, 1.0]
GreenFunction.Φ₁(x, Yε_der, Yε_der_2nd)
GreenFunction.Φ₂(x, Yε_der, Yε_der_2nd)


A = reshape(collect(1.0:16), (4, 4))
# A = collect(1.0:16);

component = 5

res1 = fft(A, 1) # [component]
ifft(res1, 1)


res = 0
for i ∈ 1:length(A)
    res += exp(-im * 2 * π * (i - 1) * (component - 1) / length(A)) * A[i]
end
res


grid_X, grid_Y = GreenFunction.gen_grid_FFT(π, 1.0, 32)

for i ∈ 1:grid_X
    println(i)
end

