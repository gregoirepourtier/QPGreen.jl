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

"""
"""
function get_F̂ⱼ(x, j₁, j₂, c̃, α, Y_der, Y_der_2nd)

    if norm(x) ≠ 0
        Φ₁(x) = (2 + ln(abs(x))) * Y_ϵ_der(abs(x)) / abs(x) + Y_ϵ_der_2nd(abs(x)) * ln(abs(x))
        Φ₂(x) = x[1] * Φ₁(x)

        Φ̂₁ⱼ = 1 # to compute via 2D FFT
        Φ̂₂ⱼ = 1 # to compute via 2D FFT

        F̂₁ⱼ = 1 / (j₁^2 + j₂^2 * π^2 / c̃^2) * (1 / (2 * √π * c̃) + 1 / (2 * π) * Φ̂₁ⱼ)
        F̂₂ⱼ = 1 / (j₁^2 + j₂^2 * π^2 / c̃^2) * (-2 * im * j₁ * F̂₁ⱼ + 1 / (2 * π) * Φ̂₂ⱼ)
    else
        integral = 1 # replace by formula
        F̂₁₀ = -1 / (2 * √(π * c̃)) * integral
        F̂₂₀ = 0
    end

    return F̂₁ⱼ, F̂₂ⱼ
end
