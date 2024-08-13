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
function get_K̂ⱼ(x, j₁, j₂, c̃, α, χ_der::T, k; degree_legendre=3) where {T}

    αⱼ₁ = α + j₁
    βⱼ₁ = abs(αⱼ₁) <= k ? √(k^2 - αⱼ₁^2) : im * √(αⱼ₁^2 - k^2)

    ξ, w = gausslegendre(degree_legendre)

    f₁(x) = exp(im * βⱼ₁ * x) * χ_der(x) * exp(-im * j₂ * π / c̃ * x)
    f₂(x) = exp(im * βⱼ₁ * x) * χ_der(x) * exp(im * j₂ * π / c̃ * x)

    integral_1 = dot(w, quad.(f₁, ξ, 0, c̃))
    integral_2 = dot(w, quad.(f₂, ξ, 0, c̃))

    return 1 / (2 * √(π * c̃)) * (1 / (αⱼ₁^2 + (j₂ * π / c̃)^2 - k^2) +
            1 / (2 * βⱼ₁ * (j₂ * π / c̃ - βⱼ₁)) * integral_1 -
            1 / (2 * βⱼ₁ * (j₂ * π / c̃ + βⱼ₁)) * integral_2)
end

function Φ₁(x, Yε_der, Yε_der_2nd)
    x_norm = norm(x)

    if x_norm ≠ 0
        return (2 + log(x_norm)) * Yε_der(x_norm) / x_norm + Yε_der_2nd(x_norm) * log(x_norm)
    end
end

Φ₂(x, Yε_der, Yε_der_2nd) = x[1] * Φ₁(x, Yε_der, Yε_der_2nd)

"""
"""
function get_F̂ⱼ(x, j₁, j₂, c̃, α, ε, Yε::T1, Yε_der::T2, Yε_der_2nd::T3, Φ̂₁ⱼ, Φ̂₂ⱼ) where {T1, T2, T3}

    if (j₁^2 + j₂^2) ≠ 0
        F̂₁ⱼ = 1 / (j₁^2 + j₂^2 * π^2 / c̃^2) * (1 / (2 * √(π * c̃)) + 1 / (2 * π) * Φ̂₁ⱼ)
        F̂₂ⱼ = 1 / (j₁^2 + j₂^2 * π^2 / c̃^2) * (-2 * im * j₁ * F̂₁ⱼ + 1 / (2 * π) * Φ̂₂ⱼ)
    else # special case |j| = 0
        ξ, w = gausslegendre(3)

        f₀(x) = x * log(x) * Yε(x)
        integral = dot(w, quad.(f₀, ξ, 0, 2 * ε))

        F̂₁ⱼ = -1 / (2 * √(π * c̃)) * integral
        F̂₂ⱼ = 0
    end

    return F̂₁ⱼ, F̂₂ⱼ
end
