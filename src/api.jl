# API of the package 

"""
"""
function fm_method_preparation(x, α, c, c̃, k, χ_der::T1, Yε::T2, Yε_der::T3, Yε_der_2nd::T4; grid_size=100,
                               ε=0.1) where {T1, T2, T3, T4}

    total_pts = 4 * grid_size^2

    # Generate the grid
    grid_X, grid_Y = gen_grid_FFT(π, c̃, grid_size)

    set_of_pt_grid = zeros(total_pts, 2)
    for i ∈ 1:total_pts
        set_of_pt_grid[i, 1] = grid_X[i]
        set_of_pt_grid[i, 2] = grid_Y[i]
    end

    # 1. Preparation step
    evaluation_Φ₁ = reshape(Φ₁(set_of_pt_grid, Yε_der, Yε_der_2nd, total_pts), (2 * grid_size, 2 * grid_size))
    evaluation_Φ₂ = reshape(Φ₂(set_of_pt_grid, Yε_der, Yε_der_2nd, total_pts), (2 * grid_size, 2 * grid_size))

    Φ̂₁ⱼ = fft(evaluation_Φ₁, 1)
    Φ̂₂ⱼ = fft(evaluation_Φ₂, 1)

    _fourier_coeffs_grid = zeros(Complex{Float64}, total_pts)
    for i ∈ 1:total_pts
        j₁ = grid_X[i]
        j₂ = grid_Y[i]

        # a) Calculate Fourier Coefficients K̂ⱼ
        K̂ⱼ = get_K̂ⱼ(x, j₁, j₂, c̃, α, χ_der, k)

        # b) Calculate Fourier Coefficients L̂ⱼ
        F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(x, j₁, j₂, c̃, α, ε, Yε, Yε_der, Yε_der_2nd, Φ̂₁ⱼ[i], Φ̂₂ⱼ[i])
        L̂ⱼ = K̂ⱼ - F̂₁ⱼ + im * α * F̂₂ⱼ

        # c) Calculate the values of Lₙ at the grid points by 2D IFFT
        _fourier_coeffs_grid[i] = L̂ⱼ
    end

    fourier_coeffs_grid = reshape(_fourier_coeffs_grid, (2 * grid_size, 2 * grid_size))

    ifft(fourier_coeffs_grid, 1)

    nothing

    ## Question : FFT -> how to precise the basis functions that we are using  in the implementation ?? -> 
    # I guess multiply the results in order to obtain the chosen basis function ?
    # _fm_method()
end


"""
"""
function fm_method_calculation(x, c, c̃)

    # 2. Calculation

    if abs(x[2]) > c
        green_function_eigfct_exp(x; k=10, α=0.3, nb_terms=100)
    else
        t = get_t(x1)

        # Bicubic Interpolation or else to get Lₙ(t, x₂)

        # Get K(t, x₂)

        # Calculate the approximate value of G(x)
    end

end
