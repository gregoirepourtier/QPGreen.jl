# FFT-based algorithm

function Φ₁(x, cache::IntegrationCache)
    return (2 + log(x)) * Yε_1st_der(x, cache) / x + Yε_2nd_der(x, cache) * log(x)
end

function ρₓ(x, x_norm, cache::IntegrationCache)
    return log(x_norm) * Yε_1st_der(x_norm, cache) * x[1]^2 / x_norm
end

function h₁_reduced(x, x_norm, α, cache::IntegrationCache)
    first_term = log(x_norm) * Yε_1st_der(x_norm, cache) * x[1] / x_norm
    third_term = log(x_norm) * Yε_1st_der(x_norm, cache) * x[1]^3 / x_norm
    return first_term - α^2 / 2 * third_term
end

function h₂_reduced(x, x_norm, α, cache::IntegrationCache)
    first_term = log(x_norm) * Yε_1st_der(x_norm, cache) * x[2] / x_norm
    second_term = log(x_norm) * Yε_1st_der(x_norm, cache) * x[1] * x[2] / x_norm
    third_term = log(x_norm) * Yε_1st_der(x_norm, cache) * x[1]^2 * x[2] / x_norm
    return first_term - im * α * second_term - α^2 / 2 * third_term
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
    get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ, Φ̂₂ⱼ, F̂₁ⱼ₀, type_α)

Calculate the Fourier coefficients F̂ⱼ.
Input arguments:

  - j₁: first index
  - j₂: second index
  - c̃: parameter of the periodicity
  - Φ̂₁ⱼ: Fourier coefficients of the function Φ₁
  - Φ̂₂ⱼ: Fourier coefficients of the function Φ₂
  - F̂₁ⱼ₀: Fourier coefficient of the function F₁ at |j| = 0
  - type_α: type of the parameter α

Returns the Fourier coefficients F̂ⱼ.
"""
function get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ, Φ̂₂ⱼ, F̂₁ⱼ₀::Complex{type_α}, ::Type{type_α}) where {type_α}
    if (j₁^2 + j₂^2) ≠ 0
        cst = j₁^2 + j₂^2 * π^2 / c̃^2
        F̂₁ⱼ = 1 / cst * (1 / (2 * √(π * c̃)) + 1 / (2 * π) * Φ̂₁ⱼ)
        F̂₂ⱼ = 1 / cst * (-2 * im * j₁ * F̂₁ⱼ + 1 / (2 * π) * Φ̂₂ⱼ)
    else # special case |j| = 0
        F̂₁ⱼ = F̂₁ⱼ₀
        F̂₂ⱼ = zero(Complex{type_α})
    end
    return F̂₁ⱼ, F̂₂ⱼ
end

"""
    get_Ĥⱼ(j₁, j₂, csts, F̂₁ⱼ, F̂₂ⱼ, ρ̂ₓⱼ, Φ̂₃ⱼ, ĥ₁ⱼ, ĥ₂ⱼ, Ĥ₁ⱼ₀, type_α)

Calculate the Fourier coefficients Ĥⱼ.
Input arguments:

    - j₁: first index
    - j₂: second index
    - csts: Named tuple of the constants for the problem definition
    - F̂₁ⱼ: Fourier coefficients of the function F₁
    - F̂₂ⱼ: Fourier coefficients of the function F₂
    - ρ̂ₓⱼ: Fourier coefficients of the function ρₓ
    - Φ̂₃ⱼ: Fourier coefficients of the function Φ₃
    - ĥ₁ⱼ: Fourier coefficients of the function h₁_reduced
    - ĥ₂ⱼ: Fourier coefficients of the function h₂_reduced
    - Ĥ₁ⱼ₀: Fourier coefficient of the function H₁ at |j| = 0
    - type_α: type of the parameter α

Returns the Fourier coefficients Ĥⱼ.
"""
function get_Ĥⱼ(j₁, j₂, csts::NamedTuple, F̂₁ⱼ, F̂₂ⱼ, ρ̂ₓⱼ, Φ̂₃ⱼ, ĥ₁ⱼ, ĥ₂ⱼ, Ĥ₁ⱼ₀::Complex{type_α},
                 ::Type{type_α}) where {type_α}

    c̃, k, α = (csts.c̃, csts.k, csts.α)

    if (j₁^2 + j₂^2) ≠ 0
        cst = 1 / (j₁^2 + j₂^2 * π^2 / c̃^2)

        j₁_F̂₂ⱼ = im * j₁ * F̂₂ⱼ
        qt2 = j₁_F̂₂ⱼ - F̂₁ⱼ + 1 / (2 * π) * ρ̂ₓⱼ

        main_fft_coeff = cst * (-2 * j₁_F̂₂ⱼ -
                                2 * qt2 +
                                1 / π * ρ̂ₓⱼ +
                                1 / (2 * π) * Φ̂₃ⱼ)

        Ĥ₁ⱼ = -k^2 / 2 * F̂₂ⱼ +
               im * j₁ * F̂₁ⱼ -
               im * α * qt2 -
               α^2 / 2 * (im * j₁ * main_fft_coeff - 2 * F̂₂ⱼ) +
               1 / (2 * π) * ĥ₁ⱼ


        Ĥ₂ⱼ = -k^2 / 2 * F̂₂ⱼ +
               im * (π / c̃) * j₂ * F̂₁ⱼ +
               α * (π / c̃) * j₂ * F̂₂ⱼ -
               α^2 / 2 * im * (π / c̃) * j₂ * main_fft_coeff +
               1 / (2 * π) * ĥ₂ⱼ

    else # special case |j| = 0
        Ĥ₁ⱼ = Ĥ₁ⱼ₀
        Ĥ₂ⱼ = zero(Complex{type_α})
    end

    return Ĥ₁ⱼ, Ĥ₂ⱼ
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
