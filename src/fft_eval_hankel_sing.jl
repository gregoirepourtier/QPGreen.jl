# FFT-based algorithm to compute quasi-periodic Green's function for 2D Helmholtz equation

"""
    fm_method_preparation_hankel(csts, grid_size)

Preparation step of the FFT-based algorithm.
Input arguments:

  - csts: tuple of constants (α, k, c, c̃, ε, order) (recommended value: 0.4341)
  - grid_size: size of the grid

Returns the Fourier coefficients of the function Lₙ.
"""
function fm_method_preparation_hankel(csts::NamedTuple, grid_size::Integer)

    α, k, c, c̃, ε, order = (csts.α, csts.k, csts.c, csts.c̃, csts.ε, csts.order)
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

    evaluation_Φ = Matrix{type_α}(undef, N, N)

    for i ∈ eachindex(xx)
        for j ∈ eachindex(yy)
            x = xx[i]
            y = yy[j]
            println(Φ(norm((x, y)), k, cache_Yε))
            evaluation_Φ[i, j] = norm((x, y)) != 0.0 ? Φ(norm((x, y)), k, cache_Yε) : zero(type_α)
        end
    end

    # display(evaluation_Φ[10:20, 10:20])

    Φ̂ⱼ = Matrix{Complex{type_α}}(undef, N ÷ 2 + 1, N)

    shift_sample_Φ = fftshift(evaluation_Φ)
    # display(shift_sample_Φ[1:10, 1:10])
    fft_Φ_eval = rfft(shift_sample_Φ)
    rfftshift_normalization!(Φ̂ⱼ, fft_Φ_eval, N, c̃)

    # println(typeof(Φ̂ⱼ))
    # display(Φ̂ⱼ[1:10, 1:10])

    p = plan_fft!(fft_cache.shift_sample_eval_int) # ; flags=FFTW.ESTIMATE, timelimit=Inf)
    @inbounds for i ∈ 1:N
        # a) Calculate Fourier Coefficients K̂ⱼ (using FFT to compute Fourier integrals)
        @views get_K̂ⱼ!(K̂ⱼ[:, i], csts, N, i, fft_cache, cache_χ, p)
        j₁ = fft_cache.j_idx[i]

        # b) Calculate Fourier Coefficients L̂ⱼ
        if i > N ÷ 2 + 1
            idx_fft_row = N - i + 2
            for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]
                # F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ[idx_fft_row, j], Φ̂₂ⱼ[idx_fft_row, j], F̂ⱼ₀, type_α)
                F̂ⱼ = get_F̂ⱼ_hankel(j₁, j₂, α, k, c̃, Φ̂ⱼ[idx_fft_row, j], type_α)
                L̂ⱼ[j, i] = K̂ⱼ[j, i] - exp(-im * α * j₁) * F̂ⱼ
            end
        else
            idx_fft_row = i
            for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]
                # F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ[idx_fft_row, j], conj(Φ̂₂ⱼ[idx_fft_row, j]), F̂ⱼ₀, type_α)
                F̂ⱼ = get_F̂ⱼ_hankel(j₁, j₂, α, k, c̃, Φ̂ⱼ[idx_fft_row, j], type_α)
                L̂ⱼ[j, i] = K̂ⱼ[j, i] - exp(-im * α * j₁) * F̂ⱼ
            end
        end
    end

    Lₙ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ))))

    # Create the Bicubic Interpolation function
    interp_cubic = cubic_spline_interpolation((xx, yy), Lₙ; extrapolation_bc=Line())
    # interp_cubic = linear_interpolation((xx, yy), Lₙ; extrapolation_bc=Line()) # More efficient but less accurate

    return interp_cubic, cache_Yε
end


"""
    fm_method_calculation_hankel(x, csts, interp_cubic, cache_Yε; nb_terms=100)

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
function fm_method_calculation_hankel(x, csts::NamedTuple, interp_cubic::T, cache_Yε::IntegrationCache; nb_terms=50) where {T}

    α, c = (csts.α, csts.c)
    k = csts.k

    if abs(x[2]) > c
        # @info "The point is outside the domain D_c"
        return eigfunc_expansion(x, csts; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = interp_cubic(t, x[2])

        # Get K(t, x₂)
        # K_t_x₂ = Lₙ_t_x₂ + f₁((t, x[2]), cache_Yε) - im * α * f₂((t, x[2]), cache_Yε)
        K_t_x₂ = Lₙ_t_x₂ + exp(-im * α * t) * f_hankel((t, x[2]), k, cache_Yε)

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
function fm_method_calculation_smooth_hankel(x, csts::NamedTuple, interp_cubic::T, cache_Yε::IntegrationCache;
                                             nb_terms=50) where {T}

    α, c, k = (csts.α, csts.c, csts.k)

    if abs(x[2]) > c
        # @info "The point is outside the domain D_c"
        return eigfunc_expansion(x, csts; nb_terms=nb_terms) - im / 4 * Bessels.hankelh1(0, k * norm(x))
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = interp_cubic(t, x[2])

        x_norm = norm(SVector(t, x[2])) # or norm(x)?
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
