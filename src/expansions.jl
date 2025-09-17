# Formulas for the α-quasi-periodic Green's function of the 2D Helmholtz equation (basic image and eigenfunction expansions)

"""
    image_expansion(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the α-quasi-periodic Green's function using the image expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the α-quasi-periodic Green's function for the 2D Helmholtz equation at the point z,
    computed via the basic image expansion.
"""
function image_expansion(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the Green function by basic image expansion
    cst_term = im / 4
    G = cst_term * Bessels.hankelh1(0, k * √(z[1]^2 + z[2]^2))
    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)
        G += cst_term * exp(im * period * α * -n) * Bessels.hankelh1(0, k * r₋ₙ) +
             cst_term * exp(im * period * α * n) * Bessels.hankelh1(0, k * rₙ)
    end

    G
end

"""
    image_expansion_gradient(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the gradient of the α-quasi-periodic Green's function using the image expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the gradient of the α-quasi-periodic Green's function for the 2D Helmholtz equation at
    the point z, computed via the basic image expansion.
"""
function image_expansion_gradient(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the derivative of the Green function by basic image expansion
    r₀ = √(z[1]^2 + z[2]^2)
    cst_term = -im / 4 * k
    common_term = cst_term * Bessels.hankelh1(1, k * r₀) / r₀
    G_prime_x1 = common_term * z[1]
    G_prime_x2 = common_term * z[2]
    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)

        common_term_1 = cst_term * exp(im * period * α * -n) * Bessels.hankelh1(1, k * r₋ₙ) / r₋ₙ
        common_term_2 = cst_term * exp(im * period * α * n) * Bessels.hankelh1(1, k * rₙ) / rₙ
        G_prime_x1 += common_term_1 * (z[1] - period * -n) + common_term_2 * (z[1] - period * n)
        G_prime_x2 += common_term_1 * z[2] + common_term_2 * z[2]
    end

    SVector(G_prime_x1, G_prime_x2)
end

"""
    image_expansion_hessian(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the Hessian of the α-quasi-periodic Green's function using the image expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the Hessian of the α-quasi-periodic Green's function for the 2D Helmholtz equation at
    the point z, computed via the basic image expansion.
"""
function image_expansion_hessian(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the second derivative of the Green function by basic image expansion
    r₀ = √(z[1]^2 + z[2]^2)
    cst_term1 = -im / 4 * k^2
    cst_term2 = -im / 4 * k
    bessel_term = Bessels.hankelh1(1, k * r₀)
    common_term = cst_term1 * (bessel_term / (k * r₀) - Bessels.hankelh1(2, k * r₀)) / r₀^2
    G_primeprime_x1x1 = common_term * z[1]^2 + cst_term2 * bessel_term * (1 / r₀ - z[1]^2 / r₀^3)
    G_primeprime_x1x2 = common_term * z[1] * z[2] + cst_term2 * bessel_term * (-z[1] * z[2] / r₀^3)
    G_primeprime_x2x2 = common_term * z[2]^2 + cst_term2 * bessel_term * (1 / r₀ - z[2]^2 / r₀^3)
    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)

        common_term_1 = cst_term1 * exp(im * period * α * -n) *
                        (Bessels.hankelh1(1, k * r₋ₙ) / (k * r₋ₙ) - Bessels.hankelh1(2, k * r₋ₙ)) / r₋ₙ^2
        common_term_2 = cst_term2 * exp(im * period * α * -n) * Bessels.hankelh1(1, k * r₋ₙ)
        common_term_3 = cst_term1 * exp(im * period * α * n) *
                        (Bessels.hankelh1(1, k * rₙ) / (k * rₙ) - Bessels.hankelh1(2, k * rₙ)) / rₙ^2
        common_term_4 = cst_term2 * exp(im * period * α * n) * Bessels.hankelh1(1, k * rₙ)

        G_primeprime_x1x1 += common_term_1 * (z[1] - period * -n)^2 +
                             common_term_2 * (1 / r₋ₙ - (z[1] - period * -n)^2 / r₋ₙ^3) +
                             common_term_3 * (z[1] - period * n)^2 +
                             common_term_4 * (1 / rₙ - (z[1] - period * n)^2 / rₙ^3)

        G_primeprime_x1x2 += common_term_1 * (z[1] - period * -n) * z[2] +
                             common_term_2 * (-(z[1] - period * -n) * z[2] / r₋ₙ^3) +
                             common_term_3 * (z[1] - period * n) * z[2] +
                             common_term_4 * (-(z[1] - period * n) * z[2] / rₙ^3)

        G_primeprime_x2x2 += common_term_1 * z[2]^2 +
                             common_term_2 * (1 / r₋ₙ - z[2]^2 / r₋ₙ^3) +
                             common_term_3 * z[2]^2 +
                             common_term_4 * (1 / rₙ - z[2]^2 / rₙ^3)
    end

    SVector(G_primeprime_x1x1, G_primeprime_x1x2, G_primeprime_x2x2)
end

"""
    image_expansion_smooth(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the smooth α-quasi-periodic Green's function (i.e. without the term ``H_0^{(1)(k|x|)}``
using the image expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the smooth α-quasi-periodic Green's function for the 2D Helmholtz equation at the point z,
    computed via the basic image expansion.
"""
function image_expansion_smooth(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the Green function by basic image expansion
    cst_term = im / 4
    G = zero(Complex{eltype(z)})
    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)
        G += cst_term * exp(im * period * α * -n) * Bessels.hankelh1(0, k * r₋ₙ) +
             cst_term * exp(im * period * α * n) * Bessels.hankelh1(0, k * rₙ)
    end

    G
end

"""
    image_expansion_gradient_smooth(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the gradient of the smooth α-quasi-periodic Green's function using the image expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the gradient of the smooth α-quasi-periodic Green's function for 2D Helmholtz equation
    at the point z, computed via the basic image expansion.
"""
function image_expansion_gradient_smooth(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the gradient of the Green function by basic image expansion
    G_prime_x1 = zero(Complex{eltype(z)})
    G_prime_x2 = zero(Complex{eltype(z)})

    cst_term = -im / 4 * k

    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)

        common_term1 = cst_term * exp(im * period * α * -n) / r₋ₙ
        common_term2 = cst_term * exp(im * period * α * n) / rₙ
        G_prime_x1 += common_term1 * Bessels.hankelh1(1, k * r₋ₙ) * (z[1] - period * -n) +
                      common_term2 * Bessels.hankelh1(1, k * rₙ) * (z[1] - period * n)
        G_prime_x2 += common_term1 * Bessels.hankelh1(1, k * r₋ₙ) * z[2] +
                      common_term2 * Bessels.hankelh1(1, k * rₙ) * z[2]
    end

    SVector(G_prime_x1, G_prime_x2)
end

"""
    image_expansion_hessian_smooth(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the Hessian of the smooth α-quasi-periodic Green's function using the image expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the Hessian of the smooth α-quasi-periodic Green's function for the 2D Helmholtz equation at
    the point z, computed via the basic image expansion.
"""
function image_expansion_hessian_smooth(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the second derivative of the Green function by basic image expansion
    cst_term1 = -im / 4 * k^2
    cst_term2 = -im / 4 * k

    G_primeprime_x1x1 = zero(Complex{eltype(z)})
    G_primeprime_x1x2 = zero(Complex{eltype(z)})
    G_primeprime_x2x2 = zero(Complex{eltype(z)})

    for n ∈ 1:nb_terms
        r₋ₙ = √((z[1] - period * -n)^2 + z[2]^2)
        rₙ = √((z[1] - period * n)^2 + z[2]^2)

        common_term_1 = cst_term1 * exp(im * period * α * -n) *
                        (Bessels.hankelh1(1, k * r₋ₙ) / (k * r₋ₙ) - Bessels.hankelh1(2, k * r₋ₙ)) / r₋ₙ^2
        common_term_2 = cst_term2 * exp(im * period * α * -n) * Bessels.hankelh1(1, k * r₋ₙ)
        common_term_3 = cst_term1 * exp(im * period * α * n) *
                        (Bessels.hankelh1(1, k * rₙ) / (k * rₙ) - Bessels.hankelh1(2, k * rₙ)) / rₙ^2
        common_term_4 = cst_term2 * exp(im * period * α * n) * Bessels.hankelh1(1, k * rₙ)

        G_primeprime_x1x1 += common_term_1 * (z[1] - period * -n)^2 +
                             common_term_2 * (1 / r₋ₙ - (z[1] - period * -n)^2 / r₋ₙ^3) +
                             common_term_3 * (z[1] - period * n)^2 +
                             common_term_4 * (1 / rₙ - (z[1] - period * n)^2 / rₙ^3)

        G_primeprime_x1x2 += common_term_1 * (z[1] - period * -n) * z[2] +
                             common_term_2 * (-(z[1] - period * -n) * z[2] / r₋ₙ^3) +
                             common_term_3 * (z[1] - period * n) * z[2] +
                             common_term_4 * (-(z[1] - period * n) * z[2] / rₙ^3)

        G_primeprime_x2x2 += common_term_1 * z[2]^2 +
                             common_term_2 * (1 / r₋ₙ - z[2]^2 / r₋ₙ^3) +
                             common_term_3 * z[2]^2 +
                             common_term_4 * (1 / rₙ - z[2]^2 / rₙ^3)
    end

    SVector(G_primeprime_x1x1, G_primeprime_x1x2, G_primeprime_x2x2)
end



"""
    eigfunc_expansion(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the α-quasi-periodic Green's function using the eigenfunction expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the α-quasi-periodic Green's function for 2D Helmholtz equation at the point z,
    computed by the basic eigenfunction expansion.
"""
function eigfunc_expansion(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the Green function by basic eigenfunction expansion
    cst_term = im / (2 * period)
    β₀ = abs(α) <= k ? √(k^2 - α^2) : im * √(α^2 - k^2)
    G = cst_term * (1 / β₀) * exp(im * α * z[1] + im * β₀ * abs(z[2]))
    for n ∈ 1:nb_terms
        α₋ₙ = α - n
        αₙ = α + n

        β₋ₙ = abs(α₋ₙ) <= k ? √(k^2 - α₋ₙ^2) : im * √(α₋ₙ^2 - k^2)
        βₙ = abs(αₙ) <= k ? √(k^2 - αₙ^2) : im * √(αₙ^2 - k^2)

        G += cst_term * (1 / β₋ₙ) * exp(im * α₋ₙ * z[1] + im * β₋ₙ * abs(z[2])) +
             cst_term * (1 / βₙ) * exp(im * αₙ * z[1] + im * βₙ * abs(z[2]))
    end

    G
end

"""
    eigfunc_expansion_gradient(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the gradient of the α-quasi-periodic Green's function using the eigenfunction expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the gradient of the α-quasi-periodic Green's function for 2D Helmholtz equation at
    the point z, computed via the basic eigenfunction expansion.
"""
function eigfunc_expansion_gradient(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the gradient of the Green function by basic eigenfunction expansion
    β₀ = abs(α) <= k ? √(k^2 - α^2) : im * √(α^2 - k^2)
    cst_term = 1 / (2 * period)
    common_term = cst_term * exp(im * α * z[1] + im * β₀ * abs(z[2]))
    G_prime_x1 = -common_term * α / β₀
    G_prime_x2 = -common_term * sign(z[2])
    for n ∈ 1:nb_terms
        α₋ₙ = α - n
        αₙ = α + n

        β₋ₙ = abs(α₋ₙ) <= k ? √(k^2 - α₋ₙ^2) : im * √(α₋ₙ^2 - k^2)
        βₙ = abs(αₙ) <= k ? √(k^2 - αₙ^2) : im * √(αₙ^2 - k^2)

        exp_term_minus = exp(im * α₋ₙ * z[1] + im * β₋ₙ * abs(z[2]))
        exp_term_plus = exp(im * αₙ * z[1] + im * βₙ * abs(z[2]))

        G_prime_x1 += -cst_term * α₋ₙ / β₋ₙ * exp_term_minus -
                      cst_term * αₙ / βₙ * exp_term_plus
        G_prime_x2 += -cst_term * sign(z[2]) * (exp_term_minus + exp_term_plus)
    end

    SVector(G_prime_x1, G_prime_x2)
end


"""
    eigfunc_expansion_hessian(z, params::NamedTuple; period=2π, nb_terms=50)

Compute the Hessian of the α-quasi-periodic Green's function using the eigenfunction expansion.

# Input arguments

  - z: coordinates of the difference between the target point and source point.
  - params: named tuple of the physical and numerical parameters for the problem definition.

# Returns

  - The value of the hessian of the α-quasi-periodic Green's function for 2D Helmholtz equation at
    the point z, computed via the basic eigenfunction expansion.
"""
function eigfunc_expansion_hessian(z, params::NamedTuple; period=2π, nb_terms=50)
    α, k = (params.alpha, params.k)

    # Compute the value of the second derivative of the Green function by basic eigenfunction expansion
    β₀ = abs(α) <= k ? √(k^2 - α^2) : im * √(α^2 - k^2)
    cst_term = im / (2 * period)
    common_term = cst_term * exp(im * α * z[1] + im * β₀ * abs(z[2]))
    G_primeprime_x1x1 = common_term * -α^2 / β₀
    G_primeprime_x1x2 = common_term * -α * sign(z[2])
    G_primeprime_x2x2 = common_term * -β₀
    for n ∈ 1:nb_terms
        α₋ₙ = α - n
        αₙ = α + n
        β₋ₙ = abs(α₋ₙ) <= k ? √(k^2 - α₋ₙ^2) : im * √(α₋ₙ^2 - k^2)
        βₙ = abs(αₙ) <= k ? √(k^2 - αₙ^2) : im * √(αₙ^2 - k^2)

        exp_term_minus = cst_term * exp(im * α₋ₙ * z[1] + im * β₋ₙ * abs(z[2]))
        exp_term_plus = cst_term * exp(im * αₙ * z[1] + im * βₙ * abs(z[2]))

        G_primeprime_x1x1 += -α₋ₙ^2 / β₋ₙ * exp_term_minus +
                             -αₙ^2 / βₙ * exp_term_plus
        G_primeprime_x1x2 += -α₋ₙ * sign(z[2]) * exp_term_minus +
                             -αₙ * sign(z[2]) * exp_term_plus
        G_primeprime_x2x2 += -β₋ₙ * exp_term_minus +
                             -βₙ * exp_term_plus
    end

    SVector(G_primeprime_x1x1, G_primeprime_x1x2, G_primeprime_x2x2)
end
