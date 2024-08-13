# API of the package 

"""
"""
function fm_method_preparation(x, alpha, c, c̃, k, χ_der::T; grid_size=100) where {T}
    Dc_x1 = (-π, π)
    Dc_x2 = (-c̃, c̃)

    # Generate the grid
    grid = gen_grid_FFT(Dc_x1[2], Dc_x2[2], grid_size)

    # 1. Preparation step
    for i ∈ 1:length(grid[1])
        j₁ = grid[1][i]
        j₂ = grid[2][i]

        # a) Calculate Fourier Coefficients K̂ⱼ
        K̂ⱼ = get_K̂ⱼ(x, j₁, j₂, c̃, alpha, χ_der, k)

        # b) Calculate Fourier Coefficients L̂ⱼ
        F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(x, j₁, j₂, c̃, alpha, χ_der, χ_der_2nd)
        L̂ⱼ = K̂ⱼ - F̂₁ⱼ + im * α * F̂₂ⱼ

        # c) Calculate the values of Lₙ at the grid points by 2D IFFT
        # Lₙ = ifft(L̂ⱼ)
    end

    nothing

    ## Question : FFT -> how to precise the basis functions that we are using  in the implementation ??
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
