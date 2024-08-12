# API of the package 

"""
"""
function fm_method_preparation(x, alpha, c, c̃; grid_size=100)

    D_c_x1 = (-π, π)
    D_c_x2 = (-c̃, c̃)

    grid = gen_grid_FFT(D_c_x1[2], D_c_x2[2], grid_size)

    # 1. Preparation step
    # a) Calculate Fourier Coefficients K̂ⱼ
    for j₁ ∈ grid[1], j₂ ∈ grid[2]
        K̂ⱼ = get_K̂ⱼ(x, j₁, j₂, c̃, alpha, χ_der)
    end

    # b) Calculate Fourier Coefficients L̂ⱼ
    for j₁ ∈ grid[1], j₂ ∈ grid[2]
        F̂₁ⱼ, F̂₂ⱼ = get_L̂_j
        L̂ⱼ = K̂ⱼ - F̂₁ⱼ + im * α * F̂₂ⱼ
    end

    # c) Calculate the values of Lₙ at the grid points by 2D IFFT





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
