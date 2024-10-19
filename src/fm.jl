# FFT-based algorithm

function Φ₁(x, cache::IntegrationCache)
    return (2 + log(x)) * Yε_1st_der(x, cache) / x + Yε_2nd_der(x, cache) * log(x)
end

f₀_F̂ⱼ(x, cache::IntegrationCache) = x * log(x) * Yε(x, cache)

"""
    get_K̂ⱼ(j₁, j₂, c̃, α, χ_der, k; degree_legendre=5)

Calculate the Fourier coefficients K̂ⱼ of the function Lₙ.
Input arguments:

  - j₁: first index
  - j₂: second index
  - c̃: parameter of the basis function
  - α: parameter of the basis function
  - χ_der: derivative of the cut-off function χ
  - k: parameter of the basis function

Keyword arguments:

      - degree_legendre: degree of the Legendre polynomial

Returns the Fourier coefficients K̂ⱼ.
"""
function get_K̂ⱼ!(K̂ⱼ, eval_int_fft_1D, t_j_fft, j_idx, c̃, α, k, N, i, cache::IntegrationCache)

    αₙ = α + j_idx[i]
    βₙ = abs(αₙ) <= k ? √(k^2 - αₙ^2) : im * √(αₙ^2 - k^2)

    # if βⱼ₁ == (j₂ * π / c̃) || βⱼ₁ == (-j₂ * π / c̃)
    #     @error "Unexpected Behaviour in get_K̂ⱼ"
    # end

    eval_int_fft_1D .= integrand_fourier_fft_1D.(t_j_fft, βₙ, Ref(cache))
    eval_int_fft_1D[1:N] .= 0
    fft_eval = transpose(fftshift(fft(fftshift(eval_int_fft_1D[1:(end - 1)])))) # add Plan FFT here
    fft_eval_flipped = reverse(fft_eval)

    integral_1 = c̃ / N * fft_eval[(N ÷ 2 + 1):(N ÷ 2 + N)]
    integral_2 = c̃ / N * fft_eval_flipped[(N ÷ 2):(N ÷ 2 + N - 1)]

    @. K̂ⱼ = 1 / (2 * √(π * c̃)) * (1 / (αₙ^2 + (j_idx * π / c̃)^2 - k^2) +
              1 / (2 * βₙ * (j_idx * π / c̃ - βₙ)) * integral_1 -
              1 / (2 * βₙ * (j_idx * π / c̃ + βₙ)) * integral_2)
end

"""
    get_F̂ⱼ(j₁, j₂, c̃, ε, Yε, Φ̂₁ⱼ, Φ̂₂ⱼ; degree_legendre=5)

Calculate the Fourier coefficients F̂ⱼ of the function Lₙ.
Input arguments:

  - j₁: first index
  - j₂: second index
  - c̃: parameter of the basis function
  - ε: parameter of the function Yε
  - Yε: cut-off function Yε
  - Φ̂₁ⱼ: Fourier coefficients of the function Φ₁
  - Φ̂₂ⱼ: Fourier coefficients of the function Φ₂

Keyword arguments:

      - degree_legendre: degree of the Legendre polynomial

Returns the Fourier coefficients F̂ⱼ.
"""
function get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ, Φ̂₂ⱼ, cache::IntegrationCache)

    if (j₁^2 + j₂^2) ≠ 0
        cst = j₁^2 + j₂^2 * π^2 / c̃^2
        F̂₁ⱼ = 1 / cst * (1 / (2 * √(π * c̃)) + 1 / (2 * π) * Φ̂₁ⱼ)
        F̂₂ⱼ = 1 / cst * (-2 * im * j₁ * F̂₁ⱼ + 1 / (2 * π) * Φ̂₂ⱼ)
    else # special case |j| = 0
        integral = quadgk(x -> f₀_F̂ⱼ(x, cache), 0.0, cache.params.b)[1]

        F̂₁ⱼ = -1 / (2 * √(π * c̃)) * integral
        F̂₂ⱼ = 0
    end

    return F̂₁ⱼ, F̂₂ⱼ
end

"""
    f₁(x, Yε)

Calculate the function f₁.
Input arguments:

  - x: point at which the function is evaluated
  - Yε: cut-off function Yε

Returns the value of the function f₁.
"""
function f₁(x, cache::IntegrationCache)

    x_norm = norm(x)

    return -1 / (2 * π) * log(x_norm) * Yε(x_norm, cache)
end

"""
    f₂(x, Yε::T)

Calculate the function f₂.
Input arguments:

  - x: point at which the function is evaluated
  - Yε: cut-off function Yε

Returns the value of the function f₂.
"""
function f₂(x, cache::IntegrationCache)
    x_norm = norm(x)

    return -1 / (2 * π) * x[1] * log(x_norm) * Yε(x_norm, cache)
end

"""
    get_t(x)

Find the value of t in the interval [-π, π[ such that x = 2nπ + t.
Input arguments:

  - x: point at which the function is evaluated

Returns the value of t.
"""
function get_t(x)

    _n = x ÷ (2 * π)
    _t = x % (2 * π)

    (n, t) = -π <= _t < π ? (_n, _t) : (_n + 1, x - 2 * (_n + 1) * π)

    @assert x == 2 * n * π + t&&-π <= t < π "Error finding t in get_t"

    t
end
