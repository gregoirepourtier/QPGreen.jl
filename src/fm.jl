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
function get_K̂ⱼ(x, j₁, j₂, c̃, α, χ_der)

    βⱼ₁ = abs(α + j₁) <= k ? √(k^2 - (α + j₁)^2) : im * √((α + j₁)^2 - k^2)

    ξ, w = gausslegendre(3)

    f₁(x) = exp(im * βⱼ₁ * x[2]) * χ_der(x[2]) * exp(-im * j₂ * π / c̃ * x[2])
    f₂(x) = exp(im * βⱼ₁ * x[2]) * χ_der(x[2]) * exp(im * j₂ * π / c̃ * x[2])

    integral_1 = dot(w, quad(f₁, ξ, 0, c̃))
    integral_2 = dot(w, quad(f₂, ξ, 0, c̃))

    return 1 / (2 * √(π * c̃)) * (1 / ((α + j₁)^2 + (j₂ * π / c̃)^2 - k^2) +
            1 / (2 * βⱼ₁ * (j₂ * π / c̃ - βⱼ₁)) * integral_1 -
            1 / (2 * βⱼ₁ * (j₂ * π / c̃ + βⱼ₁)) * integral_2)
end

"""
"""
function get_L̂_j()

    Φ̂₁ⱼ = 1
    Φ̂₂ⱼ = 1

    F̂₁ⱼ = 1 / (j₁^2 + j₂^2 * π^2 / c̃^2) * (1 / (2 * √π * c̃) + 1 / (2 * π) * Φ̂₁ⱼ)
    F̂₂ⱼ = 1 / (j₁^2 + j₂^2 * π^2 / c̃^2) * (-2 * im * j₁ * F̂₁ⱼ + 1 / (2 * π) * Φ̂₂ⱼ)

    return F̂₁ⱼ, F̂₂ⱼ
end
