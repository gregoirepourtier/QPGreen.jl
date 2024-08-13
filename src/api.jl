# API of the package 

"""
"""
function fm_method_preparation(x, α, c, c̃, k, χ_der::T1, Yε::T2, Yε_der::T3, Yε_der_2nd::T4; grid_size=100,
                               ε=0.1) where {T1, T2, T3, T4}
    Dc_x1 = (-π, π)
    Dc_x2 = (-c̃, c̃)

    # Generate the grid
    grid = gen_grid_FFT(Dc_x1[2], Dc_x2[2], grid_size)
    set_of_pt_grid = Vector{Float64}[]
    for i ∈ 1:length(grid[1])
        push!(set_of_pt_grid, [grid[1][i], grid[2][i]])
    end

    evaluation_Φ₁ = Φ₁.(set_of_pt_grid, Yε_der, Yε_der_2nd)
    evaluation_Φ₂ = Φ₂.(set_of_pt_grid, Yε_der, Yε_der_2nd)

    Φ̂₁ⱼ = fft(evaluation_Φ₁, 1)
    Φ̂₂ⱼ = fft(evaluation_Φ₂, 1)

    fourier_coeffs_grid = zeros(Complex{Float64}, length(grid[1]))

    # 1. Preparation step
    for i ∈ 1:length(grid[1])
        j₁ = grid[1][i]
        j₂ = grid[2][i]

        # a) Calculate Fourier Coefficients K̂ⱼ
        K̂ⱼ = get_K̂ⱼ(x, j₁, j₂, c̃, α, χ_der, k)

        # b) Calculate Fourier Coefficients L̂ⱼ
        F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(x, j₁, j₂, c̃, α, ε, Yε, Yε_der, Yε_der_2nd, Φ̂₁ⱼ[i], Φ̂₂ⱼ[i])
        L̂ⱼ = K̂ⱼ - F̂₁ⱼ + im * α * F̂₂ⱼ

        # c) Calculate the values of Lₙ at the grid points by 2D IFFT
        fourier_coeffs_grid[i] = L̂ⱼ
    end

    iff(reshape(fourier_coeffs_grid, (grid_size, grid_size)), 1)

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
