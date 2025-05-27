# API to compute the α quasi-periodic Green's function for the 2D Helmholtz equation using the FFT-based algorithm from [Zhang2018](@cite).

"""
    init_qp_green_fft(params::NamedTuple, grid_size::Integer; derivative=false)

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

  - `derivative`: if `true`, computes additionally the first-order derivative (`∇G`) of the quasi periodic Green's function.

# Returns

    - If `derivative=false`: a NamedTuple with fields
            + `value`: Spline interpolator for the function `Ln`.
            + `cache`: Precomputed integration cache for reuse in later computations.
    - If `derivative=true`: a NamedTuple with fields
            + `value`: Spline interpolator for the function `Ln`.
            + `grad`: Tuple of spline interpolators for the first derivatives of `Ln` (`∂/∂x₁`, `∂/∂x₂`).
            + `cache`: Precomputed integration cache for reuse in later computations.
"""
function init_qp_green_fft(params::NamedTuple, grid_size::Integer; derivative=false)
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

    Φ₁_eval = Matrix{T}(undef, N, N)
    Φ₂_eval = similar(Φ₁_eval)

    # Initialize derivative-specific arrays
    L̂ⱼ₁ = derivative ? similar(L̂ⱼ) : nothing
    L̂ⱼ₂ = derivative ? similar(L̂ⱼ) : nothing
    Φ₃_eval = derivative ? similar(Φ₁_eval) : nothing
    h₁_reduced_eval = derivative ? similar(Φ₁_eval) : nothing
    h₂_reduced_eval = derivative ? similar(L̂ⱼ) : nothing
    ρₓ_eval = derivative ? similar(Φ₁_eval) : nothing

    if derivative
        @inbounds @batch for i ∈ axes(y_grid, 1), j ∈ axes(x_grid, 1)
            pt = SVector(x_grid[i], y_grid[j])
            r = norm(pt)
            if iszero(r)
                z = zero(T)
                Φ₁_eval[i, j] = z
                h₁_reduced_eval[i, j] = z
                h₂_reduced_eval[i, j] = z
                ρₓ_eval[i, j] = z
            else
                Φ₁_eval[i, j] = Φ₁(r, Yε_cache)
                h₁_reduced_eval[i, j] = h₁_reduced(pt, r, α, Yε_cache)
                h₂_reduced_eval[i, j] = h₂_reduced(pt, r, α, Yε_cache)
                ρₓ_eval[i, j] = ρₓ(pt, r, Yε_cache)
            end
        end
    else # No preparation for 1st order derivative
        @inbounds @batch for i ∈ axes(y_grid, 1), j ∈ axes(x_grid, 1)
            pt = SVector(x_grid[i], y_grid[j])
            r = norm(pt)
            Φ₁_eval[i, j] = iszero(r) ? zero(T) : Φ₁(r, Yε_cache)
        end
    end
    Φ₂_eval .= x_grid .* Φ₁_eval

    derivative && (Φ₃_eval .= x_grid .* Φ₂_eval)

    # Preallocate frequency-domain matrices (Hermitian-symmetric due to rfft)
    Φ̂₁_freq = Matrix{Complex{T}}(undef, N ÷ 2 + 1, N)
    Φ̂₂_freq = similar(Φ̂₁_freq)
    Φ̂₃_freq = derivative ? similar(Φ̂₁_freq) : nothing

    ĥ₁ⱼ = derivative ? Matrix{Complex{T}}(undef, N, N) : nothing
    ĥ₂ⱼ = derivative ? similar(ĥ₁ⱼ) : nothing

    ρ̂ₓⱼ = derivative ? Matrix{Complex{T}}(undef, N, N) : nothing

    ## Transform to frequency domain with proper normalization
    # Shift spatial samples to FFT convention
    Φ₁_shifted = fftshift(Φ₁_eval)
    Φ₂_shifted = fftshift(Φ₂_eval)

    if derivative
        Φ₃_shifted = fftshift(Φ₃_eval)
        rfftshift_normalization!(Φ̂₃_freq, rfft(Φ₃_shifted), N, c̃)

        ĥ₁ⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(h₁_reduced_eval)))
        ĥ₂ⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(h₂_reduced_eval)))

        ρ̂ₓⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(ρₓ_eval)))
    end

    # Compute real FFTs
    rfftshift_normalization!(Φ̂₁_freq, rfft(Φ₁_shifted), N, c̃)
    rfftshift_normalization!(Φ̂₂_freq, rfft(Φ₂_shifted), N, c̃)

    # Precompute FFT plan (reused for each column)
    fft_plan = plan_fft!(fft_cache.shift_sample_eval_int)

    # Precompute the integrals for |j| = 0
    F̂₀ = Complex{T}(-1 / (2 * √(π * c̃)) *
                     quadgk(x -> f₀_F̂ⱼ(x, Yε_cache), 0.0, params_Yε.b)[1])
    Ĥ₁ⱼ₀ = derivative ? im * α / (4 * √(π * c̃)) * quadgk(x -> f₀_Ĥⱼ(x, Yε_cache), 0.0, params_Yε.b)[1] : nothing

    # Process each frequency component
    if derivative
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, N, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            # Compute L̂ⱼ, L̂ⱼ₁, L̂ⱼ₂ coefficients
            @inbounds @batch for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]
                Φ̂₁ = Φ̂₁_freq[freq_idx, j]
                Φ̂₂ = use_conj ? Φ̂₂_freq[freq_idx, j] : conj(Φ̂₂_freq[freq_idx, j])
                Φ̂₃ = use_conj ? Φ̂₃_freq[freq_idx, j] : conj(Φ̂₃_freq[freq_idx, j])

                F̂₁, F̂₂ = get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁, Φ̂₂, F̂₀, T)
                Ĥ₁ⱼ, Ĥ₂ⱼ = get_Ĥⱼ(j₁, j₂, params, F̂₁, F̂₂, ρ̂ₓⱼ[i, j], Φ̂₃, ĥ₁ⱼ[i, j], ĥ₂ⱼ[i, j], Ĥ₁ⱼ₀, T)

                L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂₁ + im * α * F̂₂
                L̂ⱼ₁[j, i] = im * (α + j₁) * K̂ⱼ[j, i] - Ĥ₁ⱼ
                L̂ⱼ₂[j, i] = im * j₂ * (π / c̃) * K̂ⱼ[j, i] - Ĥ₂ⱼ
            end
        end
    else
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, N, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            # Compute L̂ⱼ coefficients
            @inbounds @batch for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]
                Φ̂₁ = Φ̂₁_freq[freq_idx, j]
                Φ̂₂ = use_conj ? Φ̂₂_freq[freq_idx, j] : conj(Φ̂₂_freq[freq_idx, j])

                F̂₁, F̂₂ = get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁, Φ̂₂, F̂₀, T)

                L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂₁ + im * α * F̂₂
            end
        end
    end

    # Transform back to spatial domain
    L_spatial = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ))))

    # Create spline interpolator
    value_interpolator = cubic_spline_interpolation((x_grid, y_grid), L_spatial; extrapolation_bc=Line())

    if derivative
        Lₙ₁ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₁))))
        Lₙ₂ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₂))))

        grad_interpolator = (∂x=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁; extrapolation_bc=Line()),
                             ∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₂; extrapolation_bc=Line()))

        return (value=value_interpolator,
                grad=grad_interpolator,
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
function eval_qp_green(x, params::NamedTuple, value_interpolator::T, Yε_cache::IntegrationCache; nb_terms=50) where {T}

    α, c = (params.alpha, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        return eigfunc_expansion(x, params; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = value_interpolator(t, x[2])

        # Get K(t, x₂)
        K_t_x₂ = Lₙ_t_x₂ + f₁((t, x[2]), Yε_cache) - im * α * f₂((t, x[2]), Yε_cache)

        # Calculate the approximate value of G(x)
        G_x = exp(im * α * x[1]) * K_t_x₂

        return G_x
    end
end


"""
    eval_smooth_qp_green(x, params::NamedTuple, value_interpolator, Yε_cache::IntegrationCache; nb_terms=50)

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
  - `Yε_cache`: Precomputed cache for cutoff function `Yε` evaluations.

# Keyword Arguments

  - `nb_terms`: Number of terms to use in the eigenfunction expansion fallback (when `x₂ ∉ [-c, c]`)

# Returns

  - `G_0`: The approximate value of the quasiperiodic Green's function at point `x`
"""
function eval_smooth_qp_green(x, params::NamedTuple, value_interpolator::T, Yε_cache::IntegrationCache;
                              nb_terms=50) where {T}

    α, c, k = (params.alpha, params.c, params.k)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        singularity = im / 4 * Bessels.hankelh1(0, k * norm(x))
        return eigfunc_expansion(x, params; nb_terms=nb_terms) - singularity
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = value_interpolator(t, x[2])

        x_norm = norm(x)
        if x_norm == 0
            G_0 = Lₙ_t_x₂ - im / 4 + 1 / (2π) * (log(k / 2) + eulergamma)
        else
            # Get K(t, x₂)
            K_t_x₂ = Lₙ_t_x₂ + f₁((t, x[2]), Yε_cache) - im * α * f₂((t, x[2]), Yε_cache)

            # Calculate the approximate value of G_0(x)
            singularity = im / 4 * Bessels.hankelh1(0, k * norm(x))
            G_0 = exp(im * α * x[1]) * K_t_x₂ - singularity
        end

        return G_0
    end
end


"""
    grad_qp_green(x, params::NamedTuple, grad::NamedTuple, Yε_cache::IntegrationCache; nb_terms=50)

Compute the first order derivative of the α-quasi-periodic Green's function ``G(x)`` using the FFT-based method [Zhang2018](@cite) with series expansion fallback.

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

  - `∇G`: The approximate value of the fist order derivative of the quasiperiodic Green's function at point `x`
"""
function grad_qp_green(x, params::NamedTuple, grad::NamedTuple{T1, T2}, Yε_cache::IntegrationCache;
                       nb_terms=50) where {T1, T2}

    α, c = (params.alpha, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        return eigfunc_expansion_derivative(x, params; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁_t_x₂ = grad.∂x(t, x[2])
        Lₙ₂_t_x₂ = grad.∂y(t, x[2])

        # Get K(t, x₂)
        K₁_t_x₂ = Lₙ₁_t_x₂ + h₁((t, x[2]), params, Yε_cache)
        K₂_t_x₂ = Lₙ₂_t_x₂ + h₂((t, x[2]), params, Yε_cache)

        # Calculate the approximate value of ∇G(x)
        grad = exp(im * α * x[1]) .* SVector(K₁_t_x₂, K₂_t_x₂)

        return grad
    end
end

"""
    grad_smooth_qp_green(x, params::NamedTuple, grad::NamedTuple, Yε_cache::IntegrationCache; nb_terms=50)

Compute the first order derivative of the smooth α-quasi-periodic Green's function using the FFT-based method [Zhang2018](@cite) with series expansion fallback.

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

  - `∇G_0`: The approximate value of the fist order derivative of the smooth quasiperiodic Green's function at point `x`
"""
function grad_smooth_qp_green(x, params::NamedTuple, grad::NamedTuple{T1, T2}, Yε_cache::IntegrationCache;
                              nb_terms=50) where {T1, T2}

    α, c, k = (params.alpha, params.c, params.k)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        singularity = im / 4 * k * Bessels.hankelh1(1, k * norm(x)) / norm(x)
        return eigfunc_expansion_derivative(x, params; nb_terms=nb_terms) +
               singularity .* x
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁_t_x₂ = grad.∂x(t, x[2])
        Lₙ₂_t_x₂ = grad.∂y(t, x[2])

        x_norm = norm(x)
        if x_norm == 0
            return SVector(Lₙ₁_t_x₂, Lₙ₂_t_x₂)
        else
            # Get K(t, x₂)
            K₁_t_x₂ = Lₙ₁_t_x₂ + h₁((t, x[2]), params, Yε_cache)
            K₂_t_x₂ = Lₙ₂_t_x₂ + h₂((t, x[2]), params, Yε_cache)

            # Calculate the approximate value of ∇G_0(x)
            singularity = im / 4 * k * Bessels.hankelh1(1, k * norm(x)) / norm(x)
            grad_G_0 = exp(im * α * x[1]) .* SVector(K₁_t_x₂, K₂_t_x₂) + singularity .* x

            return grad_G_0
        end
    end
end
