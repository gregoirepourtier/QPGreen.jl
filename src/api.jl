# API to compute the α quasi-periodic Green's function for the 2D Helmholtz equation using the FFT-based algorithm from [Zhang2018](@cite).
"""
    init_qp_green_fft(params::NamedTuple, grid_size::Integer; grad=false, hess=false)

Preparation step of the FFT-based algorithm.

# Input arguments

  - `params`: Physical and numerical parameters, containing:

      + `alpha`: Quasiperiodicity coefficient.
      + `k`: Wave number.
      + `c`: Lower cutoff parameter for function χ.
      + `c_tilde`: Upper cutoff parameter for function χ.
      + `epsilon`: cutoff parameter for function Yε (recommended: `0.4341`).
      + `order`: Quadrature order for integration.

  - `grid_size`: Number of grid points per dimension (grid is `2 * grid_size × 2 * grid_size`).

# Keyword Arguments

  - `grad`: if `true`, computes additionally the gradient `∇G` of the quasi periodic Green's function.
  - `hess`: if `true`, computes additionally the Hessian `HG` of the quasi periodic Green's function.

# Returns

    - A NamedTuple with fields
            + `value`: Spline interpolator for the function `Ln`.
            + `grad`: Tuple of spline interpolators for the first derivatives of `Ln` (`∂/∂x₁`, `∂/∂x₂`), if `grad=true`.
            + `hess`: Tuple of spline interpolators for the second derivatives of `Ln` (`∂²/∂x₁²`, `∂²/∂x₁∂x₂`, `∂²/∂x₂²`), if `hess=true`.
            + `cache`: Precomputed integration cache for reuse in later computations.
"""
function init_qp_green_fft(params::NamedTuple, grid_size::Integer; grad=false, hess=false)
    α, k, c, c̃, ε, order = (params.alpha, params.k, params.c, params.c_tilde, params.epsilon, params.order)
    c₁, c₂ = c, (c + c̃) / 2
    T = typeof(α)

    # Check that βₙ ≠ 0, i.e. √(k^2 - αₙ^2) ≠ 0 to ensure that the eigenfunction expansion is well-defined
    check_compatibility(α, k)

    # Parameters for the cutoff functions
    params_χ = IntegrationParameters(c₁, c₂, order)
    params_Yε = IntegrationParameters(ε, 2ε, order)

    # Generate caches for the cutoff functions
    χ_cache = IntegrationCache(params_χ)
    Yε_cache = IntegrationCache(params_Yε)

    # Generate the grid
    N = 2 * grid_size
    x_grid = range(-π, π - π / grid_size; length=N)
    y_grid = range(-c̃, c̃ - c̃ / grid_size; length=N)

    # Preallocate FFT workspace and matrices for FFT sample points
    fft_cache = FFTCache(N, grid_size, c̃, T)

    K̂ⱼ = Matrix{Complex{T}}(undef, N, N)
    L̂ⱼ = similar(K̂ⱼ)

    Φ_eval = Matrix{Complex{T}}(undef, N, N)

    # Initialize derivative-specific arrays
    g_eval_sing_x1 = grad ? similar(Φ_eval) : nothing
    g_eval_sing_x2 = grad ? similar(Φ_eval) : nothing
    h_eval_sing_x1x1 = hess ? similar(Φ_eval) : nothing
    h_eval_sing_x1x2 = hess ? similar(Φ_eval) : nothing
    h_eval_sing_x2x2 = hess ? similar(Φ_eval) : nothing
    L̂ⱼ₁ = grad ? similar(L̂ⱼ) : nothing
    L̂ⱼ₂ = grad ? similar(L̂ⱼ) : nothing
    L̂ⱼ₁₁ = hess ? similar(L̂ⱼ) : nothing
    L̂ⱼ₁₂ = hess ? similar(L̂ⱼ) : nothing
    L̂ⱼ₂₂ = hess ? similar(L̂ⱼ) : nothing

    if hess
        @inbounds @batch for i ∈ axes(x_grid, 1), j ∈ axes(y_grid, 1)
            pt = SVector(x_grid[i], y_grid[j])
            r = norm(pt)
            Φ_eval[i, j] = iszero(r) ? zero(Complex{T}) : Φ(r, k, Yε_cache)
            g_eval_sing_x1[i, j] = iszero(r) ? zero(Complex{T}) : g_sing(r, k, Yε_cache)
            g_eval_sing_x2[i, j] = g_eval_sing_x1[i, j]
            h_eval_sing_x1x1[i, j] = iszero(r) ? zero(Complex{T}) : h_sing(r, k, Yε_cache)
            h_eval_sing_x1x2[i, j] = h_eval_sing_x1x1[i, j]
            h_eval_sing_x2x2[i, j] = h_eval_sing_x1x1[i, j]
        end
    elseif grad && !hess
        @inbounds @batch for i ∈ axes(x_grid, 1), j ∈ axes(y_grid, 1)
            pt = SVector(x_grid[i], y_grid[j])
            r = norm(pt)
            Φ_eval[i, j] = iszero(r) ? zero(Complex{T}) : Φ(r, k, Yε_cache)
            g_eval_sing_x1[i, j] = iszero(r) ? zero(Complex{T}) : g_sing(r, k, Yε_cache)
            g_eval_sing_x2[i, j] = g_eval_sing_x1[i, j]
        end
    else
        @inbounds @batch for i ∈ axes(x_grid, 1), j ∈ axes(y_grid, 1)
            pt = SVector(x_grid[i], y_grid[j])
            r = norm(pt)
            Φ_eval[i, j] = iszero(r) ? zero(Complex{T}) : Φ(r, k, Yε_cache)
        end
    end
    Φ_eval .*= exp.(-im * α .* x_grid)

    # Preallocate frequency-domain matrices (Hermitian-symmetric due to rfft)
    Φ̂_freq = Matrix{Complex{T}}(undef, N, N)
    Ĝ_x1_freq = grad ? similar(Φ̂_freq) : nothing
    Ĝ_x2_freq = grad ? similar(Φ̂_freq) : nothing
    Ĥ_x1x1_freq = hess ? similar(Φ̂_freq) : nothing
    Ĥ_x1x2_freq = hess ? similar(Φ̂_freq) : nothing
    Ĥ_x2x2_freq = hess ? similar(Φ̂_freq) : nothing

    ## Transform to frequency domain with proper normalization
    # Shift spatial samples to FFT convention
    Φ̂_freq .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(Φ_eval)))
    if grad || hess
        g_eval_sing_x1 .*= exp.(-im * α .* x_grid) .* x_grid
        g_eval_sing_x2 .*= exp.(-im * α .* x_grid) * y_grid'
        Ĝ_x1_freq .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(g_eval_sing_x1)))
        Ĝ_x2_freq .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(g_eval_sing_x2)))
        if hess
            h_eval_sing_x1x1 .*= exp.(-im * α .* x_grid) .* (x_grid .^ 2)
            h_eval_sing_x1x2 .*= exp.(-im * α .* x_grid) .* (x_grid * y_grid')
            h_eval_sing_x2x2 .*= exp.(-im * α .* x_grid) .* (y_grid' .^ 2)
            Ĥ_x1x1_freq .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(h_eval_sing_x1x1)))
            Ĥ_x1x2_freq .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(h_eval_sing_x1x2)))
            Ĥ_x2x2_freq .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(h_eval_sing_x2x2)))
        end
    end

    # Precompute FFT plan (reused for each column)
    fft_plan = plan_fft!(fft_cache.shift_sample_eval_int)

    # Process each frequency component
    if hess
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, N, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            # Compute L̂ⱼ, L̂ⱼ₁, L̂ⱼ₂, L̂ⱼ₁₁, L̂ⱼ₁₂, L̂ⱼ₂₂ coefficients
            @inbounds @batch for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]

                cst = (α + j₁)^2 + j₂^2 * π^2 / c̃^2 - k^2
                F̂ⱼ = -1 / cst * (-1 / (2 * √(π * c̃)) + im / 4 * Φ̂_freq[i, j])

                L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂ⱼ

                Ĝⱼ₁ = im * (α + j₁) * F̂ⱼ - Ĝ_x1_freq[i, j]
                Ĝⱼ₂ = im * j₂ * (π / c̃) * F̂ⱼ - Ĝ_x2_freq[i, j]
                L̂ⱼ₁[j, i] = im * (α + j₁) * K̂ⱼ[j, i] - Ĝⱼ₁
                L̂ⱼ₂[j, i] = im * j₂ * (π / c̃) * K̂ⱼ[j, i] - Ĝⱼ₂

                L̂ⱼ₁₁[j, i] = -(α + j₁)^2 * K̂ⱼ[j, i] - im * (α + j₁) * Ĝⱼ₁ + Ĥ_x1x1_freq[i, j]
                L̂ⱼ₁₂[j, i] = -(α + j₁) * j₂ * (π / c̃) * K̂ⱼ[j, i] - im * j₂ * (π / c̃) * Ĝⱼ₁ + Ĥ_x1x2_freq[i, j]
                L̂ⱼ₂₂[j, i] = -(j₂ * π / c̃)^2 * K̂ⱼ[j, i] - im * j₂ * (π / c̃) * Ĝⱼ₂ + Ĥ_x2x2_freq[i, j]
            end
        end
    elseif grad && !hess
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, N, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            # Compute L̂ⱼ, L̂ⱼ₁, L̂ⱼ₂ coefficients
            @inbounds @batch for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]

                cst = (α + j₁)^2 + j₂^2 * π^2 / c̃^2 - k^2
                F̂ⱼ = -1 / cst * (-1 / (2 * √(π * c̃)) + im / 4 * Φ̂_freq[i, j])

                L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂ⱼ

                L̂ⱼ₁[j, i] = im * (α + j₁) * K̂ⱼ[j, i] - im * (α + j₁) * F̂ⱼ + Ĝ_x1_freq[i, j]
                L̂ⱼ₂[j, i] = im * j₂ * (π / c̃) * K̂ⱼ[j, i] - im * j₂ * (π / c̃) * F̂ⱼ + Ĝ_x2_freq[i, j]
            end
        end
    else
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, N, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            # Compute L̂ⱼ coefficients
            @inbounds @batch for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]

                cst = (α + j₁)^2 + j₂^2 * π^2 / c̃^2 - k^2
                F̂ⱼ = -1 / cst * (-1 / (2 * √(π * c̃)) + im / 4 * Φ̂_freq[i, j])

                L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂ⱼ
            end
        end
    end

    # Transform back to spatial domain
    L_spatial = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ))))

    # Create spline interpolator
    value_interpolator = cubic_spline_interpolation((x_grid, y_grid), L_spatial; extrapolation_bc=Line())

    if grad && hess
        Lₙ₁ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₁))))
        Lₙ₂ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₂))))

        grad_interpolator = (∂x=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁; extrapolation_bc=Line()),
                             ∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₂; extrapolation_bc=Line()))

        Lₙ₁₁ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₁₁))))
        Lₙ₁₂ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₁₂))))
        Lₙ₂₂ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₂₂))))

        hess_interpolator = (∂x∂x=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁₁; extrapolation_bc=Line()),
                             ∂x∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁₂; extrapolation_bc=Line()),
                             ∂y∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₂₂; extrapolation_bc=Line()))

        return (value=value_interpolator,
                grad=grad_interpolator,
                hess=hess_interpolator,
                cache=Yε_cache)
    elseif grad && !hess
        Lₙ₁ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₁))))
        Lₙ₂ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₂))))

        grad_interpolator = (∂x=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁; extrapolation_bc=Line()),
                             ∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₂; extrapolation_bc=Line()))
        return (value=value_interpolator,
                grad=grad_interpolator,
                cache=Yε_cache)
    elseif !grad && hess
        Lₙ₁₁ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₁₁))))
        Lₙ₁₂ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₁₂))))
        Lₙ₂₂ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₂₂))))

        hess_interpolator = (∂x∂x=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁₁; extrapolation_bc=Line()),
                             ∂x∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁₂; extrapolation_bc=Line()),
                             ∂y∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₂₂; extrapolation_bc=Line()))
        return (value=value_interpolator,
                hess=hess_interpolator,
                cache=Yε_cache)
    end

    return (value=value_interpolator,
            cache=Yε_cache)
end



"""
    eval_qp_green(x, params::NamedTuple, interpolator, Yε_cache::IntegrationCache; nb_terms=50)

Compute the quasiperiodic Green's function ``G(x)`` using the FFT-based method [Zhang2018](@cite) with series expansion fallback.

# Input arguments

  - `x`: 2D point at which to evaluate the Green's function.

  - `params`: Physical and numerical parameters containing:

      + `alpha`: Quasiperiodicity coefficient
      + `k`: Wavenumber
      + `c`: Lower cutoff parameter for function `χ`.
      + `c_tilde`: Upper cutoff parameter for function `χ`.
      + `epsilon`: cutoff parameter for function `Yε` (recommended: `0.4341`).
      + `order`: Quadrature order.
  - `value_interpolator`: Bicubic spline interpolator of `Ln`.
  - `Yε_cache`: Precomputed cache for cutoff function `Yε` evaluations.

# Keyword Arguments

  - `nb_terms`: Number of terms to use in the eigenfunction expansion fallback (when `x₂ ∉ [-c, c]`)

# Returns

  - `G`: The approximate value of the quasiperiodic Green's function at point `x`
"""
function eval_qp_green(x, params::NamedTuple, value_interpolator::T, Yε_cache::IntegrationCache; nb_terms=10) where {T}

    α, k, c = (params.alpha, params.k, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        return eigfunc_expansion(x, params; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = value_interpolator(t, x[2])

        x_norm = norm((t, x[2]))

        # Get K(t, x₂)
        if x_norm <= Yε_cache.params.a
            K_t_x₂ = Lₙ_t_x₂
            return exp(im * α * x[1]) * (K_t_x₂ + exp(-im * α * t) * im / 4 * hankelh1(0, k * x_norm))
        elseif x_norm >= Yε_cache.params.b
            return exp(im * α * x[1]) * Lₙ_t_x₂
        else
            sing = f_hankel(x_norm, k, Yε_cache)
            K_t_x₂ = Lₙ_t_x₂ + exp(-im * α * t) * sing
            return exp(im * α * x[1]) * K_t_x₂
        end
    end
end

"""
    eval_smooth_qp_green(x, params::NamedTuple, value_interpolator; nb_terms=50)

Compute the smooth α-quasi-periodic Green's function ``G_0(x)`` (i.e. without the term ``H_0^{(1)(k|x|)}`` using the FFT-based method [Zhang2018](@cite) with series expansion fallback.

# Input arguments

  - `x`: 2D point at which to evaluate the Green's function.

  - `params`: Physical and numerical parameters containing:

      + `alpha`: Quasiperiodicity coefficient
      + `k`: Wavenumber
      + `c`: Lower cutoff parameter for function `χ`.
      + `c_tilde`: Upper cutoff parameter for function `χ`.
      + `epsilon`: cutoff parameter for function `Yε` (recommended: `0.4341`).
      + `order`: Quadrature order
  - `value_interpolator`: Bicubic spline interpolator for `Ln`.

# Keyword Arguments

  - `nb_terms`: Number of terms to use in the eigenfunction expansion fallback (when `x₂ ∉ [-c, c]`)

# Returns

  - `G_0`: The approximate value of the quasiperiodic Green's function at point `x`
"""
function eval_smooth_qp_green(x, params::NamedTuple, value_interpolator::T, Yε_cache::IntegrationCache; nb_terms=10) where {T}

    α, c, k = (params.alpha, params.c, params.k)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        x_norm = norm(x)
        singularity = im / 4 * hankelh1(0, k * x_norm)
        return eigfunc_expansion(x, params; nb_terms=nb_terms) - singularity
    else
        t = get_t(x[1])

        x_norm = norm((t, x[2]))

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = value_interpolator(t, x[2])

        if x_norm <= Yε_cache.params.a
            if t == x[1]
                return exp(im * α * x[1]) * Lₙ_t_x₂
            else
                bessel_term_1 = im / 4 * hankelh1(0, k * x_norm)
                K_t_x₂ = Lₙ_t_x₂ + exp(-im * α * t) * bessel_term_1
                bessel_term_2 = im / 4 * hankelh1(0, k * norm(x))
                return exp(im * α * x[1]) * K_t_x₂ - bessel_term_2
            end
        elseif x_norm >= Yε_cache.params.b
            return exp(im * α * x[1]) * Lₙ_t_x₂ - im / 4 * hankelh1(0, k * norm(x))
        else
            sing = f_hankel(x_norm, k, Yε_cache)
            K_t_x₂ = Lₙ_t_x₂ + exp(-im * α * t) * sing
            bessel_term = im / 4 * hankelh1(0, k * norm(x))
            return exp(im * α * x[1]) * K_t_x₂ - bessel_term
        end
    end
end

"""
    grad_qp_green(x, params::NamedTuple, grad::NamedTuple, Yε_cache::IntegrationCache; nb_terms=50)

Compute the gradient of the α-quasi-periodic Green's function ``G(x)`` using the FFT-based method [Zhang2018](@cite) with series expansion fallback.

# Input arguments

  - `x`: 2D point at which to evaluate the Green's function.

  - `params`: Physical and numerical parameters containing:

      + `alpha`: Quasiperiodicity coefficient
      + `k`: Wavenumber
      + `c`: Lower cutoff parameter for function `χ`.
      + `c_tilde`: Upper cutoff parameter for function `χ`.
      + `epsilon`: cutoff parameter for function `Yε` (recommended: `0.4341`).
      + `order`: Quadrature order
  - `grad`: Bicubic spline interpolator for the gradient of `Ln`.
  - `Yε_cache`: Precomputed cache for cutoff function `Yε` evaluations.

# Keyword Arguments

  - `nb_terms`: Number of terms to use in the eigenfunction expansion fallback (when `x₂ ∉ [-c, c]`)

# Returns

  - `∇G`: The approximate value of the gradient of the quasiperiodic Green's function at point `x`
"""
function grad_qp_green(x, params::NamedTuple, grad::NamedTuple{T1, T2}, Yε_cache::IntegrationCache;
                       nb_terms=10) where {T1, T2}

    α, k, c = (params.alpha, params.k, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        return eigfunc_expansion_grad(x, params; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        x_norm = norm((t, x[2]))

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁_t_x₂ = grad.∂x(t, x[2])
        Lₙ₂_t_x₂ = grad.∂y(t, x[2])

        if x_norm <= Yε_cache.params.a
            common_term = exp(-im * α * t) * -im * k / 4 * hankelh1(1, k * x_norm) / x_norm
            sing_x1 = common_term * t
            sing_x2 = common_term * x[2]

            return exp(im * α * x[1]) .* SVector(Lₙ₁_t_x₂ + sing_x1, Lₙ₂_t_x₂ + sing_x2)
        elseif x_norm >= Yε_cache.params.b
            return exp(im * α * x[1]) .* SVector(Lₙ₁_t_x₂, Lₙ₂_t_x₂)
        else
            sing = grad_f_hankel(x_norm, k, Yε_cache)
            exp_term = exp(-im * α * t)
            K₁_t_x₂ = Lₙ₁_t_x₂ + exp_term * sing * t
            K₂_t_x₂ = Lₙ₂_t_x₂ + exp_term * sing * x[2]

            # Calculate the approximate value of ∇G(x)
            grad = exp(im * α * x[1]) .* SVector(K₁_t_x₂, K₂_t_x₂)

            return grad
        end
    end
end

"""
    grad_smooth_qp_green(x, params::NamedTuple, grad::NamedTuple; nb_terms=50)

Compute the gradient of the smooth α-quasi-periodic Green's function using the FFT-based method [Zhang2018](@cite) with series expansion fallback.

# Input arguments

  - `x`: 2D point at which to evaluate the Green's function.

  - `params`: Physical and numerical parameters containing:

      + `alpha`: Quasiperiodicity coefficient
      + `k`: Wavenumber
      + `c`: Lower cutoff parameter for function `χ`.
      + `c_tilde`: Upper cutoff parameter for function `χ`.
      + `epsilon`: cutoff parameter for function `Yε` (recommended: `0.4341`).
      + `order`: Quadrature order
  - `grad`: Bicubic spline interpolator for the gradient of `Ln`.

# Keyword Arguments

  - `nb_terms`: Number of terms to use in the eigenfunction expansion fallback (when `x₂ ∉ [-c, c]`)

# Returns

  - `∇G_0`: The approximate value of the gradient of the smooth quasiperiodic Green's function at point `x`
"""
function grad_smooth_qp_green(x, params::NamedTuple, grad::NamedTuple{T1, T2}, Yε_cache::IntegrationCache;
                              nb_terms=10) where {T1, T2}

    α, k, c = (params.alpha, params.k, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        x_norm = norm(x)

        singularity = im / 4 * k * hankelh1(1, k * x_norm) / x_norm
        return eigfunc_expansion_grad(x, params; nb_terms=nb_terms) + singularity .* SVector(x)
    else
        t = get_t(x[1])

        x_norm = norm((t, x[2]))

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁_t_x₂ = grad.∂x(t, x[2])
        Lₙ₂_t_x₂ = grad.∂y(t, x[2])

        if x_norm <= Yε_cache.params.a
            if t == x[1]
                return exp(im * α * x[1]) .* SVector(Lₙ₁_t_x₂, Lₙ₂_t_x₂)
            else
                bessel_term = im / 4 * k * hankelh1(1, k * x_norm) / x_norm
                common_term = exp(-im * α * t) * -bessel_term
                sing_x1 = common_term * t
                sing_x2 = common_term * x[2]

                x_norm_2 = norm(x)
                bessel_term2 = im / 4 * k * hankelh1(1, k * x_norm_2) / x_norm_2

                return exp(im * α * x[1]) .* SVector(Lₙ₁_t_x₂ + sing_x1, Lₙ₂_t_x₂ + sing_x2) + bessel_term2 .* SVector(x)
            end
        elseif x_norm >= Yε_cache.params.b
            r = norm(x)
            singularity = im / 4 * k * hankelh1(1, k * r) / r
            return exp(im * α * x[1]) .* SVector(Lₙ₁_t_x₂, Lₙ₂_t_x₂) + singularity .* SVector(x)
        else
            sing = grad_f_hankel(x_norm, k, Yε_cache)
            exp_term = exp(-im * α * t)
            K₁_t_x₂ = Lₙ₁_t_x₂ + exp_term * sing * t
            K₂_t_x₂ = Lₙ₂_t_x₂ + exp_term * sing * x[2]

            r = norm(x)
            bessel_term = -im / 4 * k * hankelh1(1, k * r) / r

            return exp(im * α * x[1]) .* SVector(K₁_t_x₂, K₂_t_x₂) - bessel_term .* SVector(x)
        end
    end
end


"""
    hess_qp_green(x, params::NamedTuple, hess::NamedTuple, Yε_cache::IntegrationCache; nb_terms=50)

Compute the Hessian of the α-quasi-periodic Green's function ``G(x)`` using the FFT-based method [Zhang2018](@cite) with series expansion fallback.

# Input arguments

  - `x`: 2D point at which to evaluate the Green's function.

  - `params`: Physical and numerical parameters containing:

      + `alpha`: Quasiperiodicity coefficient
      + `k`: Wavenumber
      + `c`: Lower cutoff parameter for function `χ`.
      + `c_tilde`: Upper cutoff parameter for function `χ`.
      + `epsilon`: cutoff parameter for function `Yε` (recommended: `0.4341`).
      + `order`: Quadrature order
  - `hess`: Bicubic spline interpolator for the Hessian of `Ln`.
  - `Yε_cache`: Precomputed cache for cutoff function `Yε` evaluations.

# Keyword Arguments

  - `nb_terms`: Number of terms to use in the eigenfunction expansion fallback (when `x₂ ∉ [-c, c]`)

# Returns

  - `HG`: The approximate value of the Hessian of the quasiperiodic Green's function at point `x`
"""
function hess_qp_green(x, params::NamedTuple, hess::NamedTuple{T1, T2}, Yε_cache::IntegrationCache;
                       nb_terms=10) where {T1, T2}

    α, k, c = (params.alpha, params.k, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        return eigfunc_expansion_hess(x, params; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁₁_t_x₂ = hess.∂x∂x(t, x[2])
        Lₙ₁₂_t_x₂ = hess.∂x∂y(t, x[2])
        Lₙ₂₂_t_x₂ = hess.∂y∂y(t, x[2])

        x_norm = norm((t, x[2]))

        if x_norm <= Yε_cache.params.a
            exp_term = exp(-im * α * t)

            (sing_x1x1, sing_x1x2, sing_x2x2) = exp_term .* singularity_hessian((t, x[2]), x_norm, k)

            return exp(im * α * x[1]) .* SVector(Lₙ₁₁_t_x₂ + sing_x1x1,
                                                 Lₙ₁₂_t_x₂ + sing_x1x2,
                                                 Lₙ₂₂_t_x₂ + sing_x2x2)
        elseif x_norm >= Yε_cache.params.b
            return exp(im * α * x[1]) .* SVector(Lₙ₁₁_t_x₂, Lₙ₁₂_t_x₂, Lₙ₂₂_t_x₂)
        else
            exp_term = exp(-im * α * t)
            (sing_x1x1, sing_x1x2, sing_x2x2) = exp_term .* hess_f_hankel((t, x[2]), x_norm, k, Yε_cache)
            K₁₁_t_x₂ = Lₙ₁₁_t_x₂ + sing_x1x1
            K₁₂_t_x₂ = Lₙ₁₂_t_x₂ + sing_x1x2
            K₂₂_t_x₂ = Lₙ₂₂_t_x₂ + sing_x2x2

            # Calculate the approximate value of HG(x)
            hess = exp(im * α * x[1]) .* SVector(K₁₁_t_x₂, K₁₂_t_x₂, K₂₂_t_x₂)

            return hess
        end
    end
end

"""
    hess_smooth_qp_green(x, params::NamedTuple, hess::NamedTuple; nb_terms=50)

Compute the Hessian of the smooth α-quasi-periodic Green's function ``G(x)`` using the FFT-based method [Zhang2018](@cite) with series expansion fallback.

# Input arguments

  - `x`: 2D point at which to evaluate the Green's function.

  - `params`: Physical and numerical parameters containing:

      + `alpha`: Quasiperiodicity coefficient
      + `k`: Wavenumber
      + `c`: Lower cutoff parameter for function `χ`.
      + `c_tilde`: Upper cutoff parameter for function `χ`.
      + `epsilon`: cutoff parameter for function `Yε` (recommended: `0.4341`).
      + `order`: Quadrature order
  - `hess`: Bicubic spline interpolator for the Hessian of `Ln`.

# Keyword Arguments

  - `nb_terms`: Number of terms to use in the eigenfunction expansion fallback (when `x₂ ∉ [-c, c]`)

# Returns

  - `HG`: The approximate value of the Hessian of the smooth quasiperiodic Green's function at point `x`
"""
function hess_smooth_qp_green(x, params::NamedTuple, hess::NamedTuple{T1, T2}, Yε_cache::IntegrationCache;
                              nb_terms=10) where {T1, T2}

    α, k, c = (params.alpha, params.k, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        x_norm = norm(x)
        singularity = singularity_hessian(x, x_norm, k)
        return eigfunc_expansion_hess(x, params; nb_terms=nb_terms) - singularity
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁₁_t_x₂ = hess.∂x∂x(t, x[2])
        Lₙ₁₂_t_x₂ = hess.∂x∂y(t, x[2])
        Lₙ₂₂_t_x₂ = hess.∂y∂y(t, x[2])

        x_norm = norm((t, x[2]))

        if x_norm <= Yε_cache.params.a
            if t == x[1]
                return exp(im * α * x[1]) .* SVector(Lₙ₁₁_t_x₂, Lₙ₁₂_t_x₂, Lₙ₂₂_t_x₂)
            else
                exp_term = exp(-im * α * t)

                (sing_x1x1, sing_x1x2, sing_x2x2) = singularity_hessian((t, x[2]), x_norm, k)

                return exp(im * α * x[1]) .* SVector(Lₙ₁₁_t_x₂ + exp_term * sing_x1x1,
                               Lₙ₁₂_t_x₂ + exp_term * sing_x1x2,
                               Lₙ₂₂_t_x₂ + exp_term * sing_x2x2) - singularity_hessian(x, norm(x), k)
            end
        elseif x_norm >= Yε_cache.params.b
            singularity = singularity_hessian(x, norm(x), k)
            return exp(im * α * x[1]) .* SVector(Lₙ₁₁_t_x₂, Lₙ₁₂_t_x₂, Lₙ₂₂_t_x₂) - singularity
        else
            exp_term = exp(-im * α * t)

            (sing_x1x1, sing_x1x2, sing_x2x2) = hess_f_hankel((t, x[2]), x_norm, k, Yε_cache)

            singularity = singularity_hessian(x, norm(x), k)

            return exp(im * α * x[1]) .* SVector(Lₙ₁₁_t_x₂ + exp_term * sing_x1x1, Lₙ₁₂_t_x₂ + exp_term * sing_x1x2,
                           Lₙ₂₂_t_x₂ + exp_term * sing_x2x2) - singularity
        end
    end
end
