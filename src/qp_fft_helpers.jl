# Helper functions for various computing Fourier coefficients.

"""
    Φ(x, k, cache::IntegrationCache)

Calculate the function Φ (removal of the singularity in the Fourier space in the case where you use
the Hankel function directly and not its asymptotic form).
"""
function Φ(x, k, cache::IntegrationCache)
    return -2 * k * Bessels.hankelh1(1, k * x) * Yε_1st_der(x, cache) +
           Bessels.hankelh1(0, k * x) * (Yε_1st_der(x, cache) / x + Yε_2nd_der(x, cache))
end

"""
    Φ₁(x, cache::IntegrationCache)

Calculate the function Φ₁ (removal of the singularity in the Fourier space).
"""
function Φ₁(x, cache::IntegrationCache)
    return (2 + log(x)) * Yε_1st_der(x, cache) / x + Yε_2nd_der(x, cache) * log(x)
end

"""
    ρₓ(x, x_norm, cache::IntegrationCache)

Calculate the function ρₓ (removal of the singularity in the Fourier space for the 1st order derivative).
"""
function ρₓ(x, x_norm, cache::IntegrationCache)
    return log(x_norm) * Yε_1st_der(x_norm, cache) * x[1]^2 / x_norm
end

"""
    h₁_reduced(x, x_norm, α, cache::IntegrationCache)

Calculate the function h₁_reduced (removal of the singularity in the Fourier space for the 1st order derivative).
"""
function h₁_reduced(x, x_norm, α, cache::IntegrationCache)
    first_term = log(x_norm) * Yε_1st_der(x_norm, cache) * x[1] / x_norm
    third_term = log(x_norm) * Yε_1st_der(x_norm, cache) * x[1]^3 / x_norm
    return first_term - α^2 / 2 * third_term
end

"""
    h₂_reduced(x, x_norm, α, cache::IntegrationCache)

Calculate the function h₂_reduced (removal of the singularity in the Fourier space for the 1st order derivative).
"""
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
    get_K̂ⱼ!(K̂ⱼ, params::NamedTuple, N, i, fft_cache::FFTCache{T}, cache::IntegrationCache, p) where {T}

Mutating function that computes the Fourier coefficients `K̂ⱼ`.

# Input arguments

  - `K̂ⱼ`: matrix to store the Fourier coefficients
  - `params`: Named tuple containing physical and numerical constants
  - `N`: size of the grid
  - `i`: index of the grid points
  - `fft_cache`: cache for the FFT
  - `cache`: cache for the cut-off function `Yε`
  - `p`: plan for the FFT

# Returns

  - The Fourier coefficients `K̂ⱼ`.
"""
function get_K̂ⱼ!(K̂ⱼ, params::NamedTuple, N, i, fft_cache::FFTCache{T}, cache::IntegrationCache, p) where {T}

    α, k, c̃ = (params.alpha, params.k, params.c_tilde)

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
    get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ, Φ̂₂ⱼ, F̂₁ⱼ₀::Complex{T}, ::Type{T}) where {T}

Calculate the Fourier coefficients `F̂ⱼ`.

# Input arguments

  - `j₁`: first index
  - `j₂`: second index
  - `c̃`: parameter of the periodicity
  - `Φ̂₁ⱼ`: Fourier coefficients of the function `Φ₁`
  - `Φ̂₂ⱼ`: Fourier coefficients of the function `Φ₂`
  - `F̂₁ⱼ₀`: Fourier coefficient of the function `F₁` at `|j| = 0`
  - `T`: type of the parameter `α`

# Returns

  - The Fourier coefficients `F̂ⱼ`.
"""
function get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ, Φ̂₂ⱼ, F̂₁ⱼ₀::Complex{T}, ::Type{T}) where {T}
    if (j₁^2 + j₂^2) ≠ 0
        cst = j₁^2 + j₂^2 * π^2 / c̃^2
        F̂₁ⱼ = 1 / cst * (1 / (2 * √(π * c̃)) + 1 / (2 * π) * Φ̂₁ⱼ)
        F̂₂ⱼ = 1 / cst * (-2 * im * j₁ * F̂₁ⱼ + 1 / (2 * π) * Φ̂₂ⱼ)
    else # special case |j| = 0
        F̂₁ⱼ = F̂₁ⱼ₀
        F̂₂ⱼ = zero(Complex{T})
    end
    return F̂₁ⱼ, F̂₂ⱼ
end

"""
    f_hankel(x, k, α, cache::IntegrationCache)

Calculate the function `f_hankel`.

# Input arguments

    - `x`: point at which the function is evaluated
    - `k`: wavenumber
    - `α`: quasi-periodicity parameter
    - `cache`: cache for the cut-off function `Yε`

# Returns

    - The value of the function `f_hankel` at the point `x`.
"""
function f_hankel(x, k, α, cache::IntegrationCache)
    x_norm = norm(x)
    return exp(-im * α * x[1]) * im / 4 * Bessels.hankelh1(0, k * x_norm) * Yε(x_norm, cache)
end

function der_f_hankel(x, k, α, cache::IntegrationCache)
    x_norm = norm(x)
    common_term = exp(-im * α * x[1]) * -im * k / 4 * Bessels.hankelh1(1, k * x_norm) / x_norm * Yε(x_norm, cache)
    return (common_term * x[1], common_term * x[2])
end

"""
    f₁(x, cache::IntegrationCache)

Calculate the function `f₁`.

# Input arguments

  - `x`: point at which the function is evaluated
  - `cache``: cache for the cut-off function `Yε`

# Returns

  - The value of the function `f₁` at the point `x`.
"""
function f₁(x, cache::IntegrationCache)

    x_norm = norm(x)

    return -1 / (2 * π) * log(x_norm) * Yε(x_norm, cache)
end

"""
    f₂(x, cache::IntegrationCache)

Calculate the function `f₂`.

# Input arguments

  - `x`: point at which the function is evaluated
  - `cache`: cache for the cut-off function `Yε`

# Returns

  - The value of the function `f₂`.
"""
function f₂(x, cache::IntegrationCache)
    x_norm = norm(x)

    return -1 / (2 * π) * x[1] * log(x_norm) * Yε(x_norm, cache)
end



"""
    get_Ĥⱼ(j₁, j₂, params::NamedTuple, F̂₁ⱼ, F̂₂ⱼ, ρ̂ₓⱼ, Φ̂₃ⱼ, ĥ₁ⱼ, ĥ₂ⱼ, Ĥ₁ⱼ₀, T)

Calculate the Fourier coefficients `Ĥⱼ`.

# Input arguments

    - `j₁`: first index
    - `j₂`: second index
    - `params`: named tuple containing physical and numerical constants
    - `F̂₁ⱼ`: Fourier coefficients of the function `F₁`
    - `F̂₂ⱼ`: Fourier coefficients of the function `F₂`
    - `ρ̂ₓⱼ`: Fourier coefficients of the function `ρₓ`
    - `Φ̂₃ⱼ`: Fourier coefficients of the function `Φ₃`
    - `ĥ₁ⱼ`: Fourier coefficients of the function `h₁_reduced`
    - `ĥ₂ⱼ`: Fourier coefficients of the function `h₂_reduced`
    - `Ĥ₁ⱼ₀`: Fourier coefficient of the function `H₁` at `|j| = 0`
    - `T`: type of the parameter `α`

# Returns

  - The Fourier coefficients `Ĥⱼ`.
"""
function get_Ĥⱼ(j₁, j₂, params::NamedTuple, F̂₁ⱼ, F̂₂ⱼ, ρ̂ₓⱼ, Φ̂₃ⱼ, ĥ₁ⱼ, ĥ₂ⱼ, Ĥ₁ⱼ₀::Complex{T},
                 ::Type{T}) where {T}

    c̃, k, α = (params.c_tilde, params.k, params.alpha)

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
        Ĥ₂ⱼ = zero(Complex{T})
    end

    return Ĥ₁ⱼ, Ĥ₂ⱼ
end

"""
    h₁(x, params::NamedTuple, cache::IntegrationCache)

Calculate the function `h₁`.

# Input arguments

  - `x`: point at which the function is evaluated
  - `params`: Named tuple containing physical and numerical constants

# Returns

  - The value of the function `h₁`.
"""
function h₁(x, params::NamedTuple, cache::IntegrationCache)
    α, k = (params.alpha, params.k)
    x_norm = norm(x)
    return -1 / (2 * π) *
           (-k^2 / 2 * x[1] * log(x_norm) + x[1] / x_norm^2 - im * α * x[1]^2 / x_norm^2 - α^2 / 2 * x[1]^3 / x_norm^2) *
           Yε(x_norm, cache)
end

"""
    h₂(x, params::NamedTuple, cache)

Calculate the function `h₂`.

# Input arguments

  - `x`: point at which the function is evaluated
  - `params`: Named tuple containing physical and numerical constants

# Returns

  - The value of the function `h₂`.
"""
function h₂(x, params::NamedTuple, cache::IntegrationCache)
    α, k = (params.alpha, params.k)
    x_norm = norm(x)
    return -1 / (2 * π) *
           (-k^2 / 2 * x[1] * log(x_norm) + x[2] / x_norm^2 - im * α * x[1] * x[2] / x_norm^2 -
            α^2 / 2 * x[1]^2 * x[2] / x_norm^2) * Yε(x_norm, cache)
end



"""
    get_t(x)

Find the value of `t` in the interval `[-π, π[` such that `x = 2nπ + t`.

# Input arguments

  - `x`: point at which the function is evaluated

# Returns

  - The value of `t`.
"""
function get_t(x)

    n = floor((x + π) / (2 * π))
    t = x - 2 * n * π

    # Ensure t is in the range [-π, π[
    if t ≥ π
        t -= 2π
        n += 1
    elseif t < -π
        t += 2π
        n -= 1
    end

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



"""
    process_frequency_component!(i, N, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

Internal helper for processing a single frequency component during quasi-periodic Green's function computation.
Handles coefficient calculations and frequency index mapping for position `i` in the FFT grid.
"""
function process_frequency_component!(i, N, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)
    j₁ = fft_cache.j_idx[i]

    # Compute K̂ⱼ coefficients
    @views get_K̂ⱼ!(K̂ⱼ[:, i], params, N, i, fft_cache, χ_cache, fft_plan)

    # Determine frequency index handling
    freq_idx = i > N ÷ 2 + 1 ? N - i + 2 : i
    use_conj = i > N ÷ 2 + 1
    return j₁, freq_idx, use_conj
end

"""
    check_compatibility(alpha, k)

Check if the parameters `alpha` and `k` are compatible, i.e., if `βₙ` is different from zero. For `βₙ`
equal to zero, the algorithm fails.
"""
function check_compatibility(alpha, k)
    n1 = -alpha + k
    n2 = -alpha - k

    if isinteger(n1) || isinteger(n2)
        error("Incompatible alpha and k: some βₙ will be zero.")
    end
end
