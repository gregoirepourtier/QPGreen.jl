# FFT-based algorithm for efficient computation of Green's functions for Helmholtz equation in 2D

# """
# """
# function _fm_method()

# end


"""
"""
fourier_basis(x, j₁, j₂, c̃) = 1 / (2 * √(π * c̃)) * exp(im * j₁ * x[1] + im * j₂ * x[2] * π / c̃)

"""
"""
function get_K̂ⱼ(j₁, j₂, c̃, α, χ_der::T, k; degree_legendre=3) where {T}

    αⱼ₁ = α + j₁
    βⱼ₁ = abs(αⱼ₁) <= k ? √(k^2 - αⱼ₁^2) : im * √(αⱼ₁^2 - k^2)

    ξ, w = gausslegendre(degree_legendre)

    f₁_K̂ⱼ(x) = exp(im * βⱼ₁ * x) * χ_der(x) * exp(-im * j₂ * π / c̃ * x)
    f₂_K̂ⱼ(x) = exp(im * βⱼ₁ * x) * χ_der(x) * exp(im * j₂ * π / c̃ * x)

    integral_1 = dot(w, quad.(f₁_K̂ⱼ, ξ, 0, c̃))
    integral_2 = dot(w, quad.(f₂_K̂ⱼ, ξ, 0, c̃))

    return 1 / (2 * √(π * c̃)) * (1 / (αⱼ₁^2 + (j₂ * π / c̃)^2 - k^2) +
            1 / (2 * βⱼ₁ * (j₂ * π / c̃ - βⱼ₁)) * integral_1 -
            1 / (2 * βⱼ₁ * (j₂ * π / c̃ + βⱼ₁)) * integral_2)
end

"""
  - x: given as a 2D array
"""
function Φ₁(x, Yε_der, Yε_der_2nd, total_pts)

    result = zeros(total_pts)
    for i ∈ 1:total_pts
        @views x_norm = norm(x[i, :])

        if x_norm ≠ 0
            result[i] = (2 + log(x_norm)) * Yε_der(x_norm) / x_norm + Yε_der_2nd(x_norm) * log(x_norm)
        end
    end

    result
end

"""
  - x: given as a 2D array
"""
Φ₂(x, Yε_der, Yε_der_2nd, total_pts) = view(x, :, 1) .* Φ₁(x, Yε_der, Yε_der_2nd, total_pts)

"""
"""
function get_F̂ⱼ(j₁, j₂, c̃, ε, Yε::T, Φ̂₁ⱼ, Φ̂₂ⱼ) where {T}

    if (j₁^2 + j₂^2) ≠ 0
        cst = j₁^2 + j₂^2 * π^2 / c̃^2
        F̂₁ⱼ = 1 / cst * (1 / (2 * √(π * c̃)) + 1 / (2 * π) * Φ̂₁ⱼ)
        F̂₂ⱼ = 1 / cst * (-2 * im * j₁ * F̂₁ⱼ + 1 / (2 * π) * Φ̂₂ⱼ)
    else # special case |j| = 0
        ξ, w = gausslegendre(3)

        f₀_F̂ⱼ(x) = x * log(x) * Yε(x)

        integral = dot(w, quad.(f₀_F̂ⱼ, ξ, 0, 2 * ε))

        F̂₁ⱼ = -1 / (2 * √(π * c̃)) * integral
        F̂₂ⱼ = 0
    end

    return F̂₁ⱼ, F̂₂ⱼ
end

"""
"""
function f₁(x, Yε::T) where {T}
    x_norm = norm(x)

    -1 / (2 * π) * log(x_norm) * Yε(x_norm)
end

"""
"""
function f₂(x, Yε::T) where {T}
    x_norm = norm(x)

    -1 / (2 * π) * x[1] * log(x_norm) * Yε(x_norm)
end

"""
"""
function get_t(x)

    _n = x[1] ÷ (2 * π)
    _t = x[1] % (2 * π)

    (n, t) = -π <= x[1] % (2 * π) < π ? (_n, _t) : (_n + 1, x[1] - 2 * (_n + 1) * π)

    @assert x[1] == 2 * n * π + t&&t >= -π && t < π "Error finding t in get_t"

    t
end
