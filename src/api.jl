# API of the package 

"""
    fm_method_preparation(csts, χ_der, Yε, Yε_der, Yε_der_2nd; grid_size=100, ε=0.1)

Returns L_n
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
    evaluation_Φ₁ = reshape(Φ₁(set_of_pt_grid, Yε_der, Yε_der_2nd, total_pts), (N, M))
    evaluation_Φ₂ = reshape(Φ₂(set_of_pt_grid, Yε_der, Yε_der_2nd, total_pts), (N, M))

    Φ̂₁ⱼ = 1 / (2 * √(π * c̃)) .* fft(evaluation_Φ₁, 1)
    Φ̂₂ⱼ = 1 / (2 * √(π * c̃)) .* fft(evaluation_Φ₂, 1)

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

    Lₙ = ifft(fourier_coeffs_grid, 1)


    return Lₙ

    ## Question : FFT -> how to precise the basis functions that we are using  in the implementation ?? -> 
    # I guess multiply the results in order to obtain the chosen basis function ?
end


"""
"""
function fm_method_calculation(x, csts, Lₙ, Yε::T; nb_terms=100) where {T}

    α, c, c̃, k = csts
    N, M = size(Lₙ)

    @assert N==M "Problem dimensions"

    # 2. Calculation

    if abs(x[2]) > c
        @info "The point is outside the domain D_c"
        evaluation_GF = green_function_eigfct_exp(x; k=10, α=0.3, nb_terms=nb_terms)
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
"""
function build_χ(x, c̃, c)

    g(x) = x^5 * (1 - x)^5
    ξ, w = gausslegendre(5)

    integral_left = dot(w, quad.(g, ξ, -(c̃ + c) / 2, -c))
    cst_left = 1 / integral_left

    integral_right = dot(w, quad.(g, ξ, c, (c̃ + c) / 2))
    cst_right = 1 / integral_right

    if abs(x) >= (c̃ + c) / 2
        return 0
    elseif abs(x) <= c
        return 1
    elseif x < -c && x > -(c̃ + c) / 2
        return cst_left * dot(w, quad.(g, ξ, -(c̃ + c) / 2, x))
    else
        x > c && x < (c̃ + c) / 2
        return cst_right * dot(w, quad.(g, ξ, x, (c̃ + c) / 2))
    end
end

"""
"""
function build_χ_der(x, c̃, c)
    g(x) = x^5 * (1 - x)^5

    ξ, w = gausslegendre(5)

    integral_left = dot(w, quad.(g, ξ, -(c̃ + c) / 2, -c))
    cst_left = 1 / integral_left

    integral_right = dot(w, quad.(g, ξ, c, (c̃ + c) / 2))
    cst_right = 1 / integral_right

    if abs(x) >= (c̃ + c) / 2
        return 0
    elseif abs(x) <= c
        return 0
    elseif x < -c && x > -(c̃ + c) / 2
        return cst_left * g.(x)
    else
        x > c && x < (c̃ + c) / 2
        return cst_right * g(x)
    end
end

"""
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
