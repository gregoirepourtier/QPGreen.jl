# API of the package 

"""
    fm_method_preparation(csts, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=100, ε=0.1)

Preparation step of the FFT-based algorithm.
Input arguments:

  - csts: tuple of the constants (α, c, c̃, k)
  - χ_der: function to build the cut-off function χ derivative
  - Yε: function to build the cut-off function Yε
  - Yε_der: function to build the derivative of the cut-off function Yε
  - Yε_der_2nd: function to build the 2nd derivative of the cut-off function Yε

Keyword arguments:

  - grid_size: size of the grid
  - ε: parameter of the function Yε

Returns the Fourier coefficients of the function Lₙ.
"""
function fm_method_preparation(csts, χ_der::T1, Yε::T2, Yε_der::T3, Yε_der_2nd::T4; grid_size=100, ε=0.1) where {T1, T2, T3, T4}

    α, c, c̃, k = csts
    total_pts = (2 * grid_size)^2 # (2N)² points

    # Generate the grid
    grid_X, grid_Y = gen_grid_FFT(π, c̃, grid_size)
    set_of_pt_grid = get_grid_pts(grid_X, grid_Y, total_pts)
    N, M = size(grid_X)

    @assert N==M==2*grid_size "Problem dimensions"

    #### 1. Preparation step ####
    evaluation_Φ₁ = transpose(reshape(Φ₁(set_of_pt_grid, Yε_der, Yε_der_2nd, total_pts), (N, M)))
    evaluation_Φ₂ = transpose(reshape(Φ₂(set_of_pt_grid, Yε_der, Yε_der_2nd, total_pts), (N, M)))

    Φ̂₁ⱼ = 1 / (2 * √(π * c̃)) .* fft(evaluation_Φ₁)
    Φ̂₂ⱼ = 1 / (2 * √(π * c̃)) .* fft(evaluation_Φ₂)
    Φ̂₁ⱼ = fftshift(Φ̂₁ⱼ)
    Φ̂₂ⱼ = fftshift(Φ̂₂ⱼ)

    fourier_coeffs_grid = zeros(Complex{Float64}, N, M)
    for i ∈ 1:N
        for j ∈ 1:M
            j₁ = grid_X[i, j]
            j₂ = grid_Y[i, j]

            # a) Calculate Fourier Coefficients K̂ⱼ
            K̂ⱼ = get_K̂ⱼ(j₁, j₂, c̃, α, χ_der, k)

            # b) Calculate Fourier Coefficients L̂ⱼ
            F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, ε, Yε, Φ̂₁ⱼ[i, j], Φ̂₂ⱼ[i, j])
            L̂ⱼ = K̂ⱼ - F̂₁ⱼ + im * α * F̂₂ⱼ

            # c) Calculate the values of Lₙ at the grid points by 2D IFFT
            fourier_coeffs_grid[i, j] = L̂ⱼ
        end
    end

    Lₙ = 1 / (2 * √(π * c̃)) .* ifft(fourier_coeffs_grid)
    # Lₙ = ifftshift(_Lₙ)

    return Lₙ
end

"""
    fm_method_calculation(x, csts, Lₙ, Yε; nb_terms=100)

Calculation step of the FFT-based algorithm.
Input arguments:

  - x: given as a 2D array
  - csts: tuple of the constants (α, c, c̃, k)
  - Lₙ: values of Lₙ at the grid points
  - Yε: function to build the cut-off function Yε

Keyword arguments:

      - nb_terms: number of terms in the series expansion

Returns the approximate value of the Green's function G(x).
"""
function fm_method_calculation(x, csts, Lₙ, Yε::T; nb_terms=100) where {T}

    α, c, c̃, k = csts
    N, M = size(Lₙ)

    @assert N==M "Problem dimensions"

    # 2. Calculation

    if abs(x[2]) > c
        @info "The point is outside the domain D_c"
        return green_function_eigfct_exp(x; k=10, α=0.3, nb_terms=nb_terms)
    else
        @info "The point is inside the domain D_c"
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        interp_cubic = cubic_spline_interpolation((range(-π, π; length=N), range(-c̃, c̃; length=N)), Lₙ)
        Lₙ_t_x₂ = interp_cubic(t, x[2])

        # Get K(t, x₂)
        K_t_x₂ = Lₙ_t_x₂ + f₁((t, x[2]), Yε) - im * α * f₂((t, x[2]), Yε)

        # Calculate the approximate value of G(x)
        G_x = exp(im * α * x[1]) * K_t_x₂

        return G_x
    end
end

"""
    build_χ(x, c̃, c)

Build the cut-off function χ.
Input arguments:

  - x: point at which the cut-off function is evaluated
  - c̃: parameter of the cut-off function
  - c: parameter of the cut-off function

Returns the value of the cut-off function at x.
"""
function build_χ(x, c̃, c)

    g(x) = x^5 * (1 - x)^5
    ξ, w = gausslegendre(5)

    if abs(x) >= (c̃ + c) / 2
        return 0
    elseif abs(x) <= c
        return 1
    elseif x < -c && x > -(c̃ + c) / 2
        integral_left = dot(w, quad.(g, ξ, -(c̃ + c) / 2, -c))
        cst_left = 1 / integral_left
        return cst_left * dot(w, quad.(g, ξ, -(c̃ + c) / 2, x))
    elseif x > c && x < (c̃ + c) / 2
        integral_right = dot(w, quad.(g, ξ, c, (c̃ + c) / 2))
        cst_right = 1 / integral_right
        return cst_right * dot(w, quad.(g, ξ, x, (c̃ + c) / 2))
    end
end

"""
    build_χ_der(x, c̃, c)

Build the derivative of the cut-off function χ.
Input arguments:

  - x: point at which the derivative of the cut-off function is evaluated
  - c̃: parameter of the cut-off function
  - c: parameter of the cut-off function

Returns the value of the derivative of the cut-off function at x.
"""
function build_χ_der(x, c̃, c)
    g(x) = x^5 * (1 - x)^5

    ξ, w = gausslegendre(5)

    if abs(x) >= (c̃ + c) / 2
        return 0
    elseif abs(x) <= c
        return 0
    elseif x < -c && x > -(c̃ + c) / 2
        integral_left = dot(w, quad.(g, ξ, -(c̃ + c) / 2, -c))
        cst_left = 1 / integral_left
        return cst_left * g.(x)
    elseif x > c && x < (c̃ + c) / 2
        integral_right = dot(w, quad.(g, ξ, c, (c̃ + c) / 2))
        cst_right = 1 / integral_right
        return cst_right * g(x)
    end
end

"""
    build_Yε(x, ε)

Build the cut-off function Yε.
Input arguments:

  - x: point at which the cut-off function is evaluated
  - ε: parameter of the cut-off function

Returns the value of the cut-off function at x.
"""
function build_Yε(x, ε)

    g(x) = x^5 * (1 - x)^5
    ξ, w = gausslegendre(5)

    integral = dot(w, quad.(g, ξ, ε, 2 * ε))
    cst = 1 / integral

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 1
    else
        return cst * dot(w, quad.(g, ξ, x, 2 * ε))
    end
end

"""
    build_Yε_der(x, ε)

Build the derivative of the cut-off function Yε.
Input arguments:

  - x: point at which the derivative of the cut-off function is evaluated
  - ε: parameter of the cut-off function

Returns the value of the derivative of the cut-off function at x.
"""
function build_Yε_der(x, ε)
    g(x) = x^5 * (1 - x)^5
    ξ, w = gausslegendre(5)

    integral = dot(w, quad.(g, ξ, ε, 2 * ε))
    cst = 1 / integral

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 0
    else
        return cst * g(x)
    end
end

"""
    build_Yε_der_2nd(x, ε)

Build the 2nd derivative of the cut-off function Yε.
Input arguments:

  - x: point at which the 2nd derivative of the cut-off function is evaluated
  - ε: parameter of the cut-off function

Returns the value of the 2nd derivative of the cut-off function at x.
"""
function build_Yε_der_2nd(x, ε)
    g(x) = -5 * (x - 1)^4 * x^4 * (2x - 1)
    ξ, w = gausslegendre(5)

    integral = dot(w, quad.(g, ξ, ε, 2 * ε))
    cst = 1 / integral

    if x >= 2 * ε
        return 0
    elseif 0 <= x <= ε
        return 0
    else
        return cst * g(x)
    end
end
