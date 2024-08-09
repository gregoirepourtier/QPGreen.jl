#%% Implementation of the α-quasi-periodic Green function for the 2D Helmholtz equation

"""
    green_function_im_exp(z; k=10, α=0.3, nb_terms=100)

  - z: coordinates of the difference between the target point and source point
  - k: wavenumber
  - α: quasi-periodicity parameter

Returns the value of the α-quasi-periodic Green function at the field point (x₁, x₂) due to a source at (y₁, y₂)
"""
function green_function_img_exp(z; k=10, α=0.3, nb_terms=100)

    G_img = zero(Complex{eltype(z)})

    range_term = nb_terms ÷ 2

    # Compute the value of the Green function by basic image expansion
    for n ∈ (-range_term):range_term
        rₙ = √((z[1] - 2 * π * n)^2 + z[2]^2)
        G_img += im / 4 * exp(im * 2 * π * α * n) * hankelh1(0, k * rₙ)
    end

    G_img
end

"""
    green_function_eigfct_exp(z; k=10, α=0.3, nb_terms=100)

  - z: coordinates of the difference between the target point and source point
  - k: wavenumber
  - α: quasi-periodicity parameter

Returns the value of the α-quasi-periodic Green function at the field point (x₁, x₂) due to a source at (y₁, y₂)
"""
function green_function_eigfct_exp(z; k=10, α=0.3, nb_terms=100)

    G_eig = zero(Complex{eltype(z)})

    range_term = nb_terms ÷ 2

    # Compute the value of the Green function by basic eigenfunction expansion
    for n ∈ (-range_term):range_term
        αₙ = α + n
        βₙ = abs(αₙ) <= k ? √(k^2 - αₙ^2) : im * √(αₙ^2 - k^2)
        G_eig += im / (4 * π) * (1 / βₙ) * exp(im * αₙ * z[1] + im * βₙ * abs(z[2]))
    end

    G_eig
end

