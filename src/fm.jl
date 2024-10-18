# FFT-based algorithm

function Φ₁(x, ε, cst_Yε, poly_Yε::T1, poly_derivative_Yε::T2) where {T1, T2}
    return (2 + log(x)) * Yε_1st_der(x, ε, cst_Yε, poly_Yε) / x + Yε_2nd_der(x, ε, cst_Yε, poly_derivative_Yε) * log(x)
end

# function get_cst(poly_Yε::T, ξ, w, ε, ::Val{order}) where {T, order}
#     eval_int_cst_Yε = SVector{order}(quad_CoV.(poly_Yε, ξ[i], ε, 2 * ε) for i ∈ 1:order)
#     integral_Yε = dot(w, eval_int_cst_Yε)
#     cst_Yε = 1 / integral_Yε
    
#     return cst_Yε
# end

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
function get_K̂ⱼ(j₁, j₂, c̃, α, k, integral_1, integral_2)

    αⱼ₁ = α + j₁
    βⱼ₁ = abs(αⱼ₁) <= k ? √(k^2 - αⱼ₁^2) : im * √(αⱼ₁^2 - k^2)

    if βⱼ₁ == (j₂ * π / c̃) || βⱼ₁ == (-j₂ * π / c̃)
        @error "Unexpected Behaviour in get_K̂ⱼ"
    end

    return 1 / (2 * √(π * c̃)) * (1 / (αⱼ₁^2 + (j₂ * π / c̃)^2 - k^2) +
            1 / (2 * βⱼ₁ * (j₂ * π / c̃ - βⱼ₁)) * integral_1 -
            1 / (2 * βⱼ₁ * (j₂ * π / c̃ + βⱼ₁)) * integral_2)
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
function get_F̂ⱼ(j₁, j₂, c̃, ε, Yε::T, Φ̂₁ⱼ, Φ̂₂ⱼ) where {T}

    if (j₁^2 + j₂^2) ≠ 0
        cst = j₁^2 + j₂^2 * π^2 / c̃^2
        F̂₁ⱼ = 1 / cst * (1 / (2 * √(π * c̃)) + 1 / (2 * π) * Φ̂₁ⱼ)
        F̂₂ⱼ = 1 / cst * (-2 * im * j₁ * F̂₁ⱼ + 1 / (2 * π) * Φ̂₂ⱼ)
    else # special case |j| = 0

        f₀_F̂ⱼ(x, p) = x * log(x) * Yε(x)

        prob = IntegralProblem(f₀_F̂ⱼ, (0.0, 2 * ε))
        integral = solve(prob, HCubatureJL()).u

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
function f₁(x, Yε::T) where {T}
    x_norm = norm(x)

    return -1 / (2 * π) * log(x_norm) * Yε(x_norm)
end

"""
    f₂(x, Yε::T)

Calculate the function f₂.
Input arguments:

  - x: point at which the function is evaluated
  - Yε: cut-off function Yε

Returns the value of the function f₂.
"""
function f₂(x, Yε::T) where {T}
    x_norm = norm(x)

    return -1 / (2 * π) * x[1] * log(x_norm) * Yε(x_norm)
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
