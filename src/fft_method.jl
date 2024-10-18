# FFT-based algorithm to compute quasi-periodic Green's function for 2D Helmholtz equation

polynomial_cutoff(x, c₁, c₂, order) = (x - c₁)^order * (x - c₂)^order

"""
    fm_method_preparation(csts; grid_size=100, ε=0.4341)

Preparation step of the FFT-based algorithm.
Input arguments:

  - csts: tuple of constants (α, c, c̃, k, order)

Keyword arguments:

  - grid_size: size of the grid
  - ε: parameter of the cut-off function Yε

Returns the Fourier coefficients of the function Lₙ.
"""
function fm_method_preparation(csts; grid_size=100, ε=0.4341)

    α, c, c̃, k, order = csts
    c₁, c₂ = (c, (c + c̃) / 2)
    ξ, w = gausslegendre(order)

    # Generate the grid
    N = 2 * grid_size
    xx = range(-π, π - π / grid_size; length=N)
    yy = range(-c̃, c̃ - c̃ / grid_size; length=N)

    poly_Yε(x) = polynomial_cutoff(x, ε, 2 * ε, order)
    poly_derivative_Yε(x) = order * (x - ε)^(order - 1) * (x - 2 * ε)^order + order * (x - ε)^order * (x - 2 * order)^(order - 1)

    eval_int_cst_Yε = quad_CoV.(poly_Yε, ξ, ε, 2 * ε)
    integral_Yε = dot(w, eval_int_cst_Yε)
    cst_Yε = 1 / integral_Yε

    #### 1. Preparation step ####
    evaluation_Φ₁ = map(x -> norm(x) != 0 ? Φ₁(norm(x), ε, cst_Yε, poly_Yε, poly_derivative_Yε) : 0.0, Iterators.product(xx, yy))

    display(evaluation_Φ₁)

    # evaluation_Φ₁ = zeros(N, N)
    # @. evaluation_Φ₁ = sqrt(xx^2 + yy'^2)
    # @. evaluation_Φ₁ = (2 + log(evaluation_Φ₁)) * Yε_1st_der(evaluation_Φ₁, ε, cst_Yε, poly_Yε) / evaluation_Φ₁ +
    #                    Yε_2nd_der(evaluation_Φ₁, ε, cst_Yε, poly_derivative_Yε) * log(evaluation_Φ₁)
    # evaluation_Φ₂ = xx .* evaluation_Φ₁

    # display(evaluation_Φ₁)

    # evaluation_Φ₁ = transpose(reshape(_evaluation_Φ₁, (N, M)))
    # evaluation_Φ₂ = transpose(reshape(_evaluation_Φ₂, (N, M)))

    # Φ̂₁ⱼ = (2 * sqrt(pi * c̃)) / (N^2) * transpose(fftshift(fft(fftshift(evaluation_Φ₁))))
    # Φ̂₂ⱼ = (2 * sqrt(pi * c̃)) / (N^2) .* transpose(fftshift(fft(fftshift(evaluation_Φ₂))))

    # t_j_fft = range(-c̃, c̃; length=(2 * N + 1))

    # f_K(x, βⱼ) = exp(im * βⱼ * x) * χ_der(x, c₁, c₂, cst_left, cst_right, poly_left, poly_right)

    # fourier_coeffs_grid = zeros(eltype(Φ̂₁ⱼ), N, N)
    # for i ∈ 1:N
    #     j₁ = -grid_size + i - 1
    #     eval_f_K = f_K(t_j_fft, beta_j1(i))
    #     eval_f_K[1:N] = 0

    # # fft_eval = transpose(fftshift(fft(fftshift(eval_f_K(1:end-1)))));
    # # fft_eval_flipped = reverse(fft_eval);

    # # integral_1 = c_tilde/N * fft_eval((N/2 + 1):(N/2 + N));
    # # integral_2 = c_tilde/N * fft_eval_flipped ((N/2):(N/2 + N - 1));s
    # # for j ∈ 1:N
    # #     j₂ = -grid_size + j - 1

    # #     # a) Calculate Fourier Coefficients K̂ⱼ
    # #     K̂ⱼ = get_K̂ⱼ(j₁, j₂, c̃, α, k)

    # #     # b) Calculate Fourier Coefficients L̂ⱼ
    # #     F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, ε, Yε, Φ̂₁ⱼ[j, i], Φ̂₂ⱼ[j, i])
    # #     L̂ⱼ = K̂ⱼ - F̂₁ⱼ + im * α * F̂₂ⱼ

    # #     # c) Calculate the values of Lₙ at the grid points by 2D IFFT
    # #     fourier_coeffs_grid[j, i] = L̂ⱼ
    # # end
    # end

    # Lₙ = M * N / (2 * √(π * c̃)) .* fftshift(ifft(fftshift(fourier_coeffs_grid)))

    # return Lₙ
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
        xs = range(-π, π - π / (N / 2); length=N)
        ys = range(-c̃, c̃ - c̃ / (N / 2); length=N)
        interp_cubic = cubic_spline_interpolation((xs, ys), Lₙ)
        Lₙ_t_x₂ = interp_cubic(t, x[2])

        # Get K(t, x₂)
        K_t_x₂ = Lₙ_t_x₂ + f₁((t, x[2]), Yε) - im * α * f₂((t, x[2]), Yε)

        # Calculate the approximate value of G(x)
        G_x = exp(im * α * x[1]) * K_t_x₂

        return G_x
    end
end
