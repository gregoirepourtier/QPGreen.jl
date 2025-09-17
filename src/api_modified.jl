# API to compute the α quasi-periodic Green's function for the 2D Helmholtz equation using the FFT-based algorithm from [Zhang2018](@cite).

function init_qp_green_fft_mod(params::NamedTuple, grid_size::Integer; gradient=false, hessian=false)
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
    L̂ⱼ₁ = gradient ? similar(L̂ⱼ) : nothing
    L̂ⱼ₂ = gradient ? similar(L̂ⱼ) : nothing
    L̂ⱼ₁₁ = hessian ? similar(L̂ⱼ) : nothing
    L̂ⱼ₁₂ = hessian ? similar(L̂ⱼ) : nothing
    L̂ⱼ₂₂ = hessian ? similar(L̂ⱼ) : nothing

    @inbounds @batch for i ∈ axes(x_grid, 1), j ∈ axes(y_grid, 1)
        pt = SVector(x_grid[i], y_grid[j])
        r = norm(pt)
        Φ_eval[i, j] = iszero(r) ? zero(Complex{T}) : Φ(r, k, Yε_cache)
    end
    Φ_eval .*= exp.(-im * α .* x_grid)

    # Preallocate frequency-domain matrices (Hermitian-symmetric due to rfft)
    Φ̂_freq = Matrix{Complex{T}}(undef, N, N)

    ## Transform to frequency domain with proper normalization
    # Shift spatial samples to FFT convention
    Φ̂_freq .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(Φ_eval)))

    # Precompute FFT plan (reused for each column)
    fft_plan = plan_fft!(fft_cache.shift_sample_eval_int)

    # Process each frequency component
    if hessian && gradient
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, N, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            # Compute L̂ⱼ, L̂ⱼ₁, L̂ⱼ₂, L̂ⱼ₁₁, L̂ⱼ₁₂, L̂ⱼ₂₂ coefficients
            @inbounds @batch for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]

                cst = (α + j₁)^2 + j₂^2 * π^2 / c̃^2 - k^2
                F̂ⱼ = -1 / cst * (-1 / (2 * √(π * c̃)) + im / 4 * Φ̂_freq[i, j])

                L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂ⱼ

                L̂ⱼ₁[j, i] = im * (α + j₁) * K̂ⱼ[j, i] - im * (α + j₁) * F̂ⱼ
                L̂ⱼ₂[j, i] = im * j₂ * (π / c̃) * K̂ⱼ[j, i] - im * j₂ * (π / c̃) * F̂ⱼ

                L̂ⱼ₁₁[j, i] = -(α + j₁)^2 * K̂ⱼ[j, i] + (α + j₁)^2 * F̂ⱼ
                L̂ⱼ₁₂[j, i] = -(α + j₁) * j₂ * (π / c̃) * K̂ⱼ[j, i] + (α + j₁) * j₂ * (π / c̃) * F̂ⱼ
                L̂ⱼ₂₂[j, i] = -(j₂ * π / c̃)^2 * K̂ⱼ[j, i] + (j₂ * π / c̃)^2 * F̂ⱼ
            end
        end
    elseif hessian && !gradient
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, N, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            # Compute L̂ⱼ, L̂ⱼ₁₁, L̂ⱼ₁₂, L̂ⱼ₂₂ coefficients
            @inbounds @batch for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]

                cst = (α + j₁)^2 + j₂^2 * π^2 / c̃^2 - k^2
                F̂ⱼ = -1 / cst * (-1 / (2 * √(π * c̃)) + im / 4 * Φ̂_freq[i, j])

                L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂ⱼ

                L̂ⱼ₁₁[j, i] = -(α + j₁)^2 * K̂ⱼ[j, i] + (α + j₁)^2 * F̂ⱼ
                L̂ⱼ₁₂[j, i] = -(α + j₁) * j₂ * (π / c̃) * K̂ⱼ[j, i] + (α + j₁) * j₂ * (π / c̃) * F̂ⱼ
                L̂ⱼ₂₂[j, i] = -(j₂ * π / c̃)^2 * K̂ⱼ[j, i] + (j₂ * π / c̃)^2 * F̂ⱼ
            end
        end
    elseif gradient && !hessian
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, N, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            # Compute L̂ⱼ, L̂ⱼ₁, L̂ⱼ₂ coefficients
            @inbounds @batch for j ∈ 1:N
                j₂ = fft_cache.j_idx[j]

                cst = (α + j₁)^2 + j₂^2 * π^2 / c̃^2 - k^2
                F̂ⱼ = -1 / cst * (-1 / (2 * √(π * c̃)) + im / 4 * Φ̂_freq[i, j])

                L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂ⱼ
                L̂ⱼ₁[j, i] = im * (α + j₁) * K̂ⱼ[j, i] - im * (α + j₁) * F̂ⱼ
                L̂ⱼ₂[j, i] = im * j₂ * (π / c̃) * K̂ⱼ[j, i] - im * j₂ * (π / c̃) * F̂ⱼ
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

    if gradient && hessian
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
    elseif gradient && !hessian
        Lₙ₁ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₁))))
        Lₙ₂ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ₂))))

        grad_interpolator = (∂x=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁; extrapolation_bc=Line()),
                             ∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₂; extrapolation_bc=Line()))
        return (value=value_interpolator,
                grad=grad_interpolator,
                cache=Yε_cache)
    elseif !gradient && hessian
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


function eval_qp_green_mod(x, params::NamedTuple, value_interpolator::T, Yε_cache::IntegrationCache; nb_terms=50) where {T}

    α, k, c = (params.alpha, params.k, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        return eigfunc_expansion(x, params; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = value_interpolator(t, x[2])

        # Get K(t, x₂)
        K_t_x₂ = Lₙ_t_x₂ + f_hankel((t, x[2]), k, α, Yε_cache)

        # Calculate the approximate value of G(x)
        G_x = exp(im * α * x[1]) * K_t_x₂

        return G_x
    end
end

function eval_smooth_qp_green_mod(x, params::NamedTuple, value_interpolator::T; nb_terms=50) where {T}

    α, c, k = (params.alpha, params.c, params.k)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        singularity = im / 4 * Bessels.hankelh1(0, k * norm(x))
        return eigfunc_expansion(x, params; nb_terms=nb_terms) - singularity
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = value_interpolator(t, x[2])

        return exp(im * α * x[1]) * Lₙ_t_x₂
    end
end

function grad_qp_green_mod(x, params::NamedTuple, grad::NamedTuple{T1, T2}, Yε_cache::IntegrationCache;
                           nb_terms=50) where {T1, T2}

    α, k, c = (params.alpha, params.k, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        return eigfunc_expansion_derivative(x, params; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁_t_x₂ = grad.∂x(t, x[2])
        Lₙ₂_t_x₂ = grad.∂y(t, x[2])

        # Get K(t, x₂)
        (sing_x1, sing_x2) = grad_f_hankel((t, x[2]), k, α, Yε_cache)
        K₁_t_x₂ = Lₙ₁_t_x₂ + sing_x1
        K₂_t_x₂ = Lₙ₂_t_x₂ + sing_x2

        # Calculate the approximate value of ∇G(x)
        grad = exp(im * α * x[1]) .* SVector(K₁_t_x₂, K₂_t_x₂)

        return grad
    end
end

function grad_smooth_qp_green_mod(x, params::NamedTuple, grad::NamedTuple{T1, T2};
                                  nb_terms=50) where {T1, T2}

    α, k, c = (params.alpha, params.k, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        singularity = im / 4 * k * Bessels.hankelh1(1, k * norm(x)) / norm(x)
        return eigfunc_expansion_gradient(x, params; nb_terms=nb_terms) + singularity .* x # check allocations
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁_t_x₂ = grad.∂x(t, x[2])
        Lₙ₂_t_x₂ = grad.∂y(t, x[2])

        return exp(im * α * x[1]) .* SVector(Lₙ₁_t_x₂, Lₙ₂_t_x₂)
    end
end

function hess_qp_green_mod(x, params::NamedTuple, hess::NamedTuple{T1, T2}, Yε_cache::IntegrationCache;
                           nb_terms=50) where {T1, T2}

    α, k, c = (params.alpha, params.k, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        return eigfunc_expansion_hessian(x, params; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁₁_t_x₂ = hess.∂x∂x(t, x[2])
        Lₙ₁₂_t_x₂ = hess.∂x∂y(t, x[2])
        Lₙ₂₂_t_x₂ = hess.∂y∂y(t, x[2])

        # Get K(t, x₂)
        (sing_x1x1, sing_x1x2, sing_x2x2) = hess_f_hankel((t, x[2]), k, α, Yε_cache)
        K₁₁_t_x₂ = Lₙ₁₁_t_x₂ + sing_x1x1
        K₁₂_t_x₂ = Lₙ₁₂_t_x₂ + sing_x1x2
        K₂₂_t_x₂ = Lₙ₂₂_t_x₂ + sing_x2x2

        # Calculate the approximate value of HG(x)
        hess = exp(im * α * x[1]) .* SVector(K₁₁_t_x₂, K₁₂_t_x₂, K₂₂_t_x₂)

        return hess
    end
end

function hess_smooth_qp_green_mod(x, params::NamedTuple, hess::NamedTuple{T1, T2};
                                  nb_terms=50) where {T1, T2}

    α, k, c = (params.alpha, params.k, params.c)

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c

        # Precompute common terms
        r = norm(Z)
        r2 = r^2
        r3 = r2 * r
        kr = k * r

        # Compute Hankel functions once
        h1 = Bessels.hankelh1(1, kr)
        h2 = Bessels.hankelh1(2, kr)

        # Precompute common factors
        common_factor1 = -im / 4 * k^2 * (h1 / kr - h2) / r2
        common_factor2 = -im / 4 * k * h1 / r3

        # Extract coordinates
        x, y = Z[1], Z[2]
        x2, y2 = x^2, y^2
        xy = x * y

        # Compute matrix elements
        m11 = common_factor1 * x2 + common_factor2 * (r2 - x2)
        m12 = common_factor1 * xy - common_factor2 * xy
        m22 = common_factor1 * y2 + common_factor2 * (r2 - y2)

        singularity = SVector(m11, m12, m22)

        return eigfunc_expansion_hessian(x, params; nb_terms=nb_terms) + singularity
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ₁₁_t_x₂ = hess.∂x∂x(t, x[2])
        Lₙ₁₂_t_x₂ = hess.∂x∂y(t, x[2])
        Lₙ₂₂_t_x₂ = hess.∂y∂y(t, x[2])

        return exp(im * α * x[1]) .* SVector(Lₙ₁₁_t_x₂, Lₙ₁₂_t_x₂, Lₙ₂₂_t_x₂)
    end
end
