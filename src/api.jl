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
    _evaluation_Φ₁ = Φ₁(set_of_pt_grid, Yε_der, Yε_der_2nd, total_pts)
    evaluation_Φ₁ = transpose(reshape(_evaluation_Φ₁, (N, M)))
    _evaluation_Φ₂ = view(set_of_pt_grid, :, 1) .* _evaluation_Φ₁
    evaluation_Φ₂ = transpose(reshape(_evaluation_Φ₂, (N, M)))

    Φ̂₁ⱼ = 1 / (2 * √(π * c̃)) .* fftshift(fft(fftshift(evaluation_Φ₁)))
    Φ̂₂ⱼ = 1 / (2 * √(π * c̃)) .* fftshift(fft(fftshift(evaluation_Φ₂)))

    fourier_coeffs_grid = zeros(eltype(Φ̂₁ⱼ), N, M)
    for i ∈ 1:N
        j₁ = -grid_size + i - 1
        for j ∈ 1:M
            j₂ = -grid_size + j - 1

            # a) Calculate Fourier Coefficients K̂ⱼ
            K̂ⱼ = get_K̂ⱼ(j₁, j₂, c̃, α, χ_der, k)

            # b) Calculate Fourier Coefficients L̂ⱼ
            F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, ε, Yε, Φ̂₁ⱼ[i, j], Φ̂₂ⱼ[i, j])
            L̂ⱼ = K̂ⱼ - F̂₁ⱼ + im * α * F̂₂ⱼ

            # c) Calculate the values of Lₙ at the grid points by 2D IFFT
            fourier_coeffs_grid[i, j] = L̂ⱼ
        end
    end

    Lₙ = (2 * √(π * c̃)) .* fftshift(ifft(fftshift(fourier_coeffs_grid)))

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
function fm_method_calculation(x, csts, Lₙ, Yε::T; α=0.3, k=10.0, nb_terms=100) where {T}

    α, c, c̃, k = csts
    N, M = size(Lₙ)

    @assert N==M "Problem dimensions"

    # 2. Calculation
    if abs(x[2]) > c
        @info "The point is outside the domain D_c"
        return green_function_eigfct_exp(x, k, α; nb_terms=nb_terms)
    else
        @info "The point is inside the domain D_c"
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        xs = range(-π, π - π / (N/2); length=N)
        ys = range(-c̃, c̃ - c̃ / (N/2); length=N)
        interp_cubic = cubic_spline_interpolation((xs, ys), Lₙ)
        Lₙ_t_x₂ = interp_cubic(t, x[2])

        # Get K(t, x₂)
        K_t_x₂ = Lₙ_t_x₂ + f₁((t, x[2]), Yε) - im * α * f₂((t, x[2]), Yε)

        # Calculate the approximate value of G(x)
        G_x = exp(im * α * x[1]) * K_t_x₂

        return G_x
    end
end
