# First Order derivatives of the Green's function

"""
    fm_method_preparation_derivative(csts, grid_size)

Preparation of the Fourier coefficients for the first order derivative of the Green's function.
"""
function fm_method_preparation_derivative(csts::NamedTuple, grid_size::Integer)

    α, c, c̃, ε, order = (csts.α, csts.c, csts.c̃, csts.ε, csts.order)
    c₁, c₂ = (c, (c + c̃) / 2)

    type_α = typeof(α)

    # Parameters for the cutoff functions
    params_χ = IntegrationParameters(c₁, c₂, order)
    params_Yε = IntegrationParameters(ε, 2 * ε, order)

    # Generate caches for the cutoff functions
    cache_χ = IntegrationCache(params_χ)
    cache_Yε = IntegrationCache(params_Yε)

    # Generate the grid
    N = 2 * grid_size
    xx = range(-π, π - π / grid_size; length=N)
    yy = range(-c̃, c̃ - c̃ / grid_size; length=N)

    fft_cache = FFT_cache(N, grid_size, csts, type_α)

    K̂ⱼ = Matrix{Complex{type_α}}(undef, N, N)
    L̂ⱼ₁ = Matrix{Complex{type_α}}(undef, N, N)
    L̂ⱼ₂ = Matrix{Complex{type_α}}(undef, N, N)

    evaluation_Φ₁ = Matrix{type_α}(undef, N, N)
    evaluation_Φ₂ = Matrix{type_α}(undef, N, N)
    evaluation_Φ₃ = Matrix{type_α}(undef, N, N)

    evaluation_h₁_reduced = Matrix{type_α}(undef, N, N)
    evaluation_h₂_reduced = Matrix{Complex{type_α}}(undef, N, N)
    evaluation_ρₓ = Matrix{type_α}(undef, N, N)

    @inbounds @batch for i ∈ eachindex(xx)
        for j ∈ eachindex(yy)
            x = xx[i]
            y = yy[j]
            eval_norm = norm((x, y))

            evaluation_Φ₁[i, j] = eval_norm != 0.0 ? Φ₁(eval_norm, cache_Yε) : zero(type_α)
            evaluation_h₁_reduced[i, j] = eval_norm != 0.0 ? h₁_reduced((x, y), eval_norm, α, cache_Yε) : zero(type_α)
            evaluation_h₂_reduced[i, j] = eval_norm != 0.0 ? h₂_reduced((x, y), eval_norm, α, cache_Yε) : zero(type_α)
            evaluation_ρₓ[i, j] = eval_norm != 0.0 ? ρₓ((x, y), eval_norm, cache_Yε) : zero(type_α)
        end
    end

    evaluation_Φ₂ .= xx .* evaluation_Φ₁
    evaluation_Φ₃ .= xx .* evaluation_Φ₂

    Φ̂₁ⱼ = Matrix{Complex{type_α}}(undef, N, N)
    Φ̂₂ⱼ = Matrix{Complex{type_α}}(undef, N, N)
    Φ̂₃ⱼ = Matrix{Complex{type_α}}(undef, N, N)

    ĥ₁ⱼ = Matrix{Complex{type_α}}(undef, N, N)
    ĥ₂ⱼ = Matrix{Complex{type_α}}(undef, N, N)

    ρ̂ₓⱼ = Matrix{Complex{type_α}}(undef, N, N)

    Φ̂₁ⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_Φ₁)))
    Φ̂₂ⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_Φ₂)))
    Φ̂₃ⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_Φ₃)))

    ĥ₁ⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_h₁_reduced)))
    ĥ₂ⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_h₂_reduced)))

    ρ̂ₓⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_ρₓ)))

    p = plan_fft!(fft_cache.shift_sample_eval_int)

    # Precompute the integrals for |j| = 0
    integral_F̂₁ⱼ₀ = quadgk(x -> f₀_F̂ⱼ(x, cache_Yε), 0.0, params_Yε.b)[1]
    F̂₁ⱼ₀ = Complex{type_α}(-1 / (2 * √(π * c̃)) * integral_F̂₁ⱼ₀)::Complex{type_α}

    integral_Ĥ₁ⱼ₀ = quadgk(x -> f₀_Ĥⱼ(x, cache_Yε), 0.0, params_Yε.b)[1]
    Ĥ₁ⱼ₀ = im * α / (4 * √(π * c̃)) * integral_Ĥ₁ⱼ₀

    @inbounds for i ∈ eachindex(xx)
        # a) Calculate Fourier Coefficients K̂ⱼ (using FFT to compute Fourier integrals)
        @views get_K̂ⱼ!(K̂ⱼ[:, i], csts, N, i, fft_cache, cache_χ, p)
        j₁ = fft_cache.j_idx[i]

        @inbounds @batch for j ∈ eachindex(yy)
            j₂ = fft_cache.j_idx[j]
            # b) Calculate Fourier Coefficients L̂ⱼ
            F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ[i, j], Φ̂₂ⱼ[i, j], F̂₁ⱼ₀, type_α)
            Ĥ₁ⱼ, Ĥ₂ⱼ = get_Ĥⱼ(j₁, j₂, csts, F̂₁ⱼ, F̂₂ⱼ, ρ̂ₓⱼ[i, j], Φ̂₃ⱼ[i, j], ĥ₁ⱼ[i, j], ĥ₂ⱼ[i, j], Ĥ₁ⱼ₀, type_α)
            L̂ⱼ₁[j, i] = im * (α + j₁) * K̂ⱼ[j, i] - Ĥ₁ⱼ
            L̂ⱼ₂[j, i] = im * j₂ * (π / c̃) * K̂ⱼ[j, i] - Ĥ₂ⱼ
        end
    end

    Lₙ₁ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₁))))
    Lₙ₂ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₂))))

    # Create the Bicubic Interpolation function
    interp_cubic_1 = cubic_spline_interpolation((xx, yy), Lₙ₁)
    interp_cubic_2 = cubic_spline_interpolation((xx, yy), Lₙ₂)
    # interp_cubic = linear_interpolation((xx, yy), Lₙ; extrapolation_bc=Line()) # More efficient but less accurate?

    return interp_cubic_1, interp_cubic_2, cache_Yε
end

"""
    fm_method_calculation_derivative(x, csts, interp_cubic_x1, interp_cubic_x2, cache_Yε; nb_terms=100)

Calculate the first order derivative of the Green's function using the FFT-based method.

Input arguments:

  - `x`: The point at which the Green's function is evaluated.
  - `csts`: A NamedTuple containing the parameters of the problem.
  - `interp_cubic_x1`: bicubic interpolation function for the first component of the Green's function.
  - `interp_cubic_x2`: bicubic interpolation function for the second component of the Green's function.
  - `cache_Yε`: IntegrationCache for the cutoff functions.

Keyword arguments:

    - `nb_terms`: Number of terms in the series expansion.

Returns the approximate value of the first order derivative of the Green's function at the point `x`.
"""
function fm_method_calculation_derivative(x, csts::NamedTuple, interp_cubic_x1::T1, interp_cubic_x2::T2,
                                          cache_Yε::IntegrationCache; nb_terms=100) where {T1, T2}

    α, c = (csts.α, csts.c)

    if abs(x[2]) > c
        return eigfunc_expansion_derivative(x, csts; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁_t_x₂ = interp_cubic_x1(t, x[2])
        Lₙ₂_t_x₂ = interp_cubic_x2(t, x[2])

        # Get K(t, x₂)
        K₁_t_x₂ = Lₙ₁_t_x₂ + h₁((t, x[2]), csts, cache_Yε)
        K₂_t_x₂ = Lₙ₂_t_x₂ + h₂((t, x[2]), csts, cache_Yε)

        # Calculate the approximate value of G(x)
        der_G_x₁ = exp(im * α * x[1]) * K₁_t_x₂
        der_G_x₂ = exp(im * α * x[1]) * K₂_t_x₂

        return (der_G_x₁, der_G_x₂)
    end
end

"""
    fm_method_calculation_derivative_smooth(x, csts, interp_cubic_x1, interp_cubic_x2, cache_Yε; nb_terms=100)

Calculate the first order derivative of the analytic part of the Green's function using the FFT-based method.

Input arguments:

  - `x`: The point at which the Green's function is evaluated.
  - `csts`: A NamedTuple containing the parameters of the problem.
  - `interp_cubic_x1`: bicubic interpolation function for the first component of the Green's function.
  - `interp_cubic_x2`: bicubic interpolation function for the second component of the Green's function.
  - `cache_Yε`: IntegrationCache for the cutoff functions.

Keyword arguments:

    - `nb_terms`: Number of terms in the series expansion.

Returns the approximate value of the first order derivative of the analytic part of the Green's function at the point `x`.
"""
function fm_method_calculation_derivative_smooth(x, csts::NamedTuple, interp_cubic_x1::T1, interp_cubic_x2::T2,
                                                 cache_Yε::IntegrationCache; nb_terms=100) where {T1, T2}

    α, c, k = (csts.α, csts.c, csts.k)

    if abs(x[2]) > c
        @info "The point is outside the domain D_c"
        return eigfunc_expansion_derivative(x, csts; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁_t_x₂ = interp_cubic_x1(t, x[2])
        Lₙ₂_t_x₂ = interp_cubic_x2(t, x[2])

        x_norm = norm(x)
        if x_norm == 0
            return (Lₙ₁_t_x₂, Lₙ₂_t_x₂)
        else
            # Get K(t, x₂)
            K₁_t_x₂ = Lₙ₁_t_x₂ + h₁((t, x[2]), csts, cache_Yε)
            K₂_t_x₂ = Lₙ₂_t_x₂ + h₂((t, x[2]), csts, cache_Yε)

            # Calculate the approximate value of G(x)
            der_G_0_x₁ = exp(im * α * x[1]) * K₁_t_x₂ + im / 4 * k * Bessels.hankelh1(1, k * x_norm) * x[1] / x_norm
            der_G_0_x₂ = exp(im * α * x[1]) * K₂_t_x₂ + im / 4 * k * Bessels.hankelh1(1, k * x_norm) * x[2] / x_norm

            return (der_G_0_x₁, der_G_0_x₂)
        end
    end
end
