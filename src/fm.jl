# FFT-based algorithm

function Φ₁(x, cache::IntegrationCache)
    return (2 + log(x)) * Yε_1st_der(x, cache) / x + Yε_2nd_der(x, cache) * log(x)
end

function Ψ₁₁(x, x_norm, cache::IntegrationCache)
    return x[1] / x_norm^2 * Yε_2nd_der(x_norm, cache) - x[1] / x_norm^3 * Yε_1st_der(x_norm, cache)
end

function Ψ₂₁(x, x_norm, cache::IntegrationCache)
    return x[2] / x_norm^2 * Yε_2nd_der(x_norm, cache) - x[2] / x_norm^3 * Yε_1st_der(x_norm, cache)
end

integrand_fourier_fft_1D(x, βⱼ, cache) = exp(im * βⱼ * x) * χ_der(x, cache)

f₀_F̂ⱼ(x, cache::IntegrationCache) = x * log(x) * Yε(x, cache)
f₀_Ĥⱼ(x, cache::IntegrationCache) = x * Yε(x, cache)

"""
    get_K̂ⱼ!((K̂ⱼ, csts, N, i, fft_cache, cache, p)

Mutating function that computes the Fourier coefficients K̂ⱼ.
Input arguments:

  - K̂ⱼ: matrix to store the Fourier coefficients
  - csts: Named tuple of the constants for the problem definition
  - N: size of the grid
  - i: index of the grid points
  - fft_cache: cache for the FFT
  - cache: cache for the cut-off function Yε
  - p: plan for the FFT

Returns the Fourier coefficients K̂ⱼ.
"""
function get_K̂ⱼ!(K̂ⱼ, csts, N, i, fft_cache::FFT_cache{T}, cache::IntegrationCache, p) where {T}

    α, k, c̃ = (csts.α, csts.k, csts.c̃)

    αₙ = α + fft_cache.j_idx[i]
    βₙ = abs(αₙ) <= k ? Complex{T}(√(k^2 - αₙ^2)) : im * √(αₙ^2 - k^2)

    fft_cache.eval_int_fft_1D .= integrand_fourier_fft_1D.(fft_cache.t_j_fft, βₙ, Ref(cache))
    fft_cache.eval_int_fft_1D[1:N] .= zero(Complex{T})
    @views fftshift!(fft_cache.shift_sample_eval_int, fft_cache.eval_int_fft_1D[1:(end - 1)])
    fft_cache.fft_eval .= p * fft_cache.shift_sample_eval_int
    fftshift!(fft_cache.shift_fft_1d, fft_cache.fft_eval)
    fft_cache.fft_eval_flipped .= transpose(fft_cache.shift_fft_1d)
    reverse!(fft_cache.fft_eval_flipped)

    @views integral_1 = fft_cache.shift_fft_1d[(N ÷ 2 + 1):(N ÷ 2 + N)]
    integral_1 .*= c̃ / N

    @views integral_2 = fft_cache.fft_eval_flipped[(N ÷ 2):(N ÷ 2 + N - 1)]
    integral_2 .*= c̃ / N

    @. K̂ⱼ = 1 / (2 * √(π * c̃)) * (1 / (αₙ^2 + (fft_cache.j_idx * π / c̃)^2 - k^2) +
              1 / (2 * βₙ * (fft_cache.j_idx * π / c̃ - βₙ)) * integral_1 -
              1 / (2 * βₙ * (fft_cache.j_idx * π / c̃ + βₙ)) * integral_2)
end

"""
    get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ, Φ̂₂ⱼ, cache, type_α)

Calculate the Fourier coefficients F̂ⱼ.
Input arguments:

  - j₁: first index
  - j₂: second index
  - c̃: parameter of the periodicity
  - Φ̂₁ⱼ: Fourier coefficients of the function Φ₁
  - Φ̂₂ⱼ: Fourier coefficients of the function Φ₂
  - cache: cache for the cut-off function Yε
  - type_α: type of the parameter α

Returns the Fourier coefficients F̂ⱼ.
"""
function get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ, Φ̂₂ⱼ, cache::IntegrationCache, ::Type{type_α}) where {type_α}
    if (j₁^2 + j₂^2) ≠ 0
        cst = j₁^2 + j₂^2 * π^2 / c̃^2
        F̂₁ⱼ = 1 / cst * (1 / (2 * √(π * c̃)) + 1 / (2 * π) * Φ̂₁ⱼ)
        F̂₂ⱼ = 1 / cst * (-2 * im * j₁ * F̂₁ⱼ + 1 / (2 * π) * Φ̂₂ⱼ)
    else # special case |j| = 0
        integral = quadgk(x -> f₀_F̂ⱼ(x, cache), 0.0, cache.params.b)[1]

        F̂₁ⱼ = Complex{type_α}(-1 / (2 * √(π * c̃)) * integral)::Complex{type_α}
        F̂₂ⱼ = zero(Complex{type_α})
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
    f₂(x, cache)

Calculate the function f₂.
Input arguments:

  - x: point at which the function is evaluated
  - cache: cache for the cut-off function Yε

Returns the value of the function f₂.
"""
function f₂(x, cache::IntegrationCache)
    x_norm = norm(x)

    return -1 / (2 * π) * x[1] * log(x_norm) * Yε(x_norm, cache)
end

"""
    h₁(x, csts, cache)

Calculate the function h₁.
Input arguments:

  - x: point at which the function is evaluated
  - csts: Named tuple of the constants for the problem definition

Returns the value of the function h₁.
"""
function h₁(x, csts, cache::IntegrationCache)
    α, k = (csts.α, csts.k)
    x_norm = norm(x)
    return -1 / (2 * π) *
           (-k^2 / 2 * x[1] * log(x_norm) + x[1] / x_norm^2 - im * α * x[1]^2 / x_norm^2 - α^2 / 2 * x[1]^3 / x_norm^2) *
           Yε(x_norm, cache)
end

"""
    h₂(x, csts, cache)

Calculate the function h₂.
Input arguments:

  - x: point at which the function is evaluated
  - csts: Named tuple of the constants for the problem definition

Returns the value of the function h₂.
"""
function h₂(x, csts, cache::IntegrationCache)
    α, k = (csts.α, csts.k)
    x_norm = norm(x)
    return -1 / (2 * π) *
           (-k^2 / 2 * x[1] * log(x_norm) + x[2] / x_norm^2 - im * α * x[1] * x[2] / x_norm^2 -
            α^2 / 2 * x[1]^2 * x[2] / x_norm^2) * Yε(x_norm, cache)
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

"""
    rfftshift_normalization!(Φ̂₁ⱼ, fft_Φ₁_eval, N, c̃)

Shift the Fourier coefficients obtained by rfft and normalize them.
"""
function rfftshift_normalization!(Φ̂₁ⱼ, fft_Φ₁_eval, N, c̃)
    circshift!(Φ̂₁ⱼ, fft_Φ₁_eval, (0, N ÷ 2))
    @views reverse!(Φ̂₁ⱼ, dims=1)
    Φ̂₁ⱼ .*= (2 * √(π * c̃)) / (N^2)
end
