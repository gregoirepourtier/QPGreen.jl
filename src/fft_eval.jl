# FFT-based algorithm to compute quasi-periodic Green's function for 2D Helmholtz equation

"""
    fm_method_preparation(csts, grid_size)

Preparation step of the FFT-based algorithm.
Input arguments:

  - csts: tuple of constants (α, k, c, c̃, ε, order) (recommended value: 0.4341)
  - grid_size: size of the grid

Returns the Fourier coefficients of the function Lₙ.
"""
function fm_method_preparation(csts::NamedTuple, grid_size::Integer)

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
    L̂ⱼ = Matrix{Complex{type_α}}(undef, N, N)

    evaluation_Φ₁ = Matrix{type_α}(undef, N, N)
    evaluation_Φ₂ = Matrix{type_α}(undef, N, N)

    @inbounds @batch for i ∈ eachindex(xx)
        for j ∈ eachindex(yy)
            x = xx[i]
            y = yy[j]
            evaluation_Φ₁[i, j] = norm((x, y)) != 0.0 ? Φ₁(norm((x, y)), cache_Yε) : zero(type_α)
        end
    end
    evaluation_Φ₂ .= xx .* evaluation_Φ₁

    Φ̂₁ⱼ = Matrix{Complex{type_α}}(undef, N ÷ 2 + 1, N)
    Φ̂₂ⱼ = Matrix{Complex{type_α}}(undef, N ÷ 2 + 1, N)

    shift_sample_Φ₁ = fftshift(evaluation_Φ₁)
    fft_Φ₁_eval = rfft(shift_sample_Φ₁)
    rfftshift_normalization!(Φ̂₁ⱼ, fft_Φ₁_eval, N, c̃)

    shift_sample_Φ₂ = fftshift(evaluation_Φ₂)
    fft_Φ₂_eval = rfft(shift_sample_Φ₂)
    rfftshift_normalization!(Φ̂₂ⱼ, fft_Φ₂_eval, N, c̃)

    p = plan_fft!(fft_cache.shift_sample_eval_int) # ; flags=FFTW.ESTIMATE, timelimit=Inf)

    integral_F̂ⱼ₀ = quadgk(x -> f₀_F̂ⱼ(x, cache_Yε), 0.0, cache_Yε.params.b)[1]
    F̂ⱼ₀ = Complex{type_α}(-1 / (2 * √(π * c̃)) * integral_F̂ⱼ₀)::Complex{type_α}

    @inbounds for i ∈ 1:N
        # a) Calculate Fourier Coefficients K̂ⱼ (using FFT to compute Fourier integrals)
        @views get_K̂ⱼ!(K̂ⱼ[:, i], csts, N, i, fft_cache, cache_χ, p)
        j₁ = fft_cache.j_idx[i]

        # b) Calculate Fourier Coefficients L̂ⱼ
        if i > N ÷ 2 + 1
            idx_fft_row = N - i + 2
            for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]
                F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ[idx_fft_row, j], Φ̂₂ⱼ[idx_fft_row, j], F̂ⱼ₀, type_α)
                L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂₁ⱼ + im * α * F̂₂ⱼ
            end
        else
            idx_fft_row = i
            for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]
                F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ[idx_fft_row, j], conj(Φ̂₂ⱼ[idx_fft_row, j]), F̂ⱼ₀, type_α)
                L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂₁ⱼ + im * α * F̂₂ⱼ
            end
        end
    end

    Lₙ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ))))

    # Create the Bicubic Interpolation function
    interp_cubic = cubic_spline_interpolation((xx, yy), Lₙ)
    # interp_cubic = linear_interpolation((xx, yy), Lₙ; extrapolation_bc=Line()) # More efficient but less accurate

    return interp_cubic, cache_Yε
end


"""
    fm_method_calculation(x, csts, interp_cubic, cache_Yε; nb_terms=100)

Calculation step of the FFT-based algorithm.
Input arguments:

  - x: given as a 2D array
  - csts: tuple of the constants (α, k, c, c̃, ε, order)
  - interp_cubic: bicubic interpolation function
  - cache_Yε: cache for the cut-off function Yε

Keyword arguments:

      - nb_terms: number of terms in the series expansion

Returns the approximate value of the Green's function G(x).
"""
function fm_method_calculation(x, csts::NamedTuple, interp_cubic::T, cache_Yε::IntegrationCache; nb_terms=100) where {T}

    α, c = (csts.α, csts.c)

    if abs(x[2]) > c
        @info "The point is outside the domain D_c"
        return eigfunc_expansion(x, csts; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = interp_cubic(t, x[2])

        # Get K(t, x₂)
        K_t_x₂ = Lₙ_t_x₂ + f₁((t, x[2]), cache_Yε) - im * α * f₂((t, x[2]), cache_Yε)

        # Calculate the approximate value of G(x)
        G_x = exp(im * α * x[1]) * K_t_x₂

        return G_x
    end
end

"""
    fm_method_calculation_smooth(x, csts, interp_cubic, cache_Yε; nb_terms=100)

Calculation step of the FFT-based algorithm for the analytic function G_0.
Input arguments:

  - x: given as a 2D array
  - csts: tuple of the constants (α, k, c, c̃, ε, order)
  - interp_cubic: bicubic interpolation function
  - cache_Yε: cache for the cut-off function Yε

Keyword arguments:

      - nb_terms: number of terms in the series expansion

Returns the approximate value of the Green's function G(x).
"""
function fm_method_calculation_smooth(x, csts::NamedTuple, interp_cubic::T, cache_Yε::IntegrationCache; nb_terms=100) where {T}

    α, c, k = (csts.α, csts.c, csts.k)

    if abs(x[2]) > c
        @info "The point is outside the domain D_c"
        return eigfunc_expansion(x, csts; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = interp_cubic(t, x[2])

        x_norm = norm(x)
        if x_norm == 0
            G_0 = Lₙ_t_x₂ - im / 4 * (1 + im * 2 / π * (log(k / 2) + eulergamma))
        else
            # Get K(t, x₂)
            K_t_x₂ = Lₙ_t_x₂ + f₁((t, x[2]), cache_Yε) - im * α * f₂((t, x[2]), cache_Yε)

            # Calculate the approximate value of G(x)
            G_0 = exp(im * α * x[1]) * K_t_x₂ - im / 4 * Bessels.hankelh1(0, k * norm(x))
        end

        return G_0
    end
end
