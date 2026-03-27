# API to compute the α quasi-periodic Green's function for the 2D Helmholtz equation using the FFT-based algorithm from [Zhang2018](@cite).

"""
    init_qp_green_fft_BIE(params::NamedTuple, grid_size::Integer; grad=false, hess=false)

Preparation step of the FFT-based algorithm with additional BIE part for new approach to replace bicubic interpolation.

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
function init_qp_green_fft_BIE(params::NamedTuple, grid_size::Union{Integer, Tuple{Integer, Integer}}; grad=false, hess=false)
    α, k, c, c̃, ε, type_cutoff = (params.alpha, params.k, params.c, params.c_tilde, params.epsilon, params.type_cutoff)
    c₁, c₂ = c, (c + c̃) / 2
    T = typeof(α)

    order = haskey(params, :order) ? params.order :
            hashkey(type_cutoff, :polynomial) == :polynomial ? error("Parameter 'order' is not defined in params") : 1

    # Check that βₙ ≠ 0, i.e. √(k^2 - αₙ^2) ≠ 0 to ensure that the eigenfunction expansion is well-defined
    check_compatibility(α, k)

    # Parameters for the cutoff functions
    params_χ = IntegrationParameters(c, c̃, order)
    params_Yε = IntegrationParameters(ε, c̃ - ε, order)

    # Generate caches for the cutoff functions
    χ_cache = IntegrationCache(params_χ, type_cutoff)
    Yε_cache = IntegrationCache(params_Yε, type_cutoff)

    # Generate the grid
    (grid_size_x, grid_size_y) = typeof(grid_size) <: Integer ? (grid_size, grid_size) : (grid_size[1], grid_size[2])
    N = 2 * grid_size_x
    M = 2 * grid_size_y
    x_grid = (-π):(π / grid_size_x):(π - π / grid_size_x + 1e-15)
    y_grid = (-c̃):(c̃ / grid_size_y):(c̃ - c̃ / grid_size_y + 1e-15)

    # Preallocate FFT workspace and matrices for FFT sample points
    fft_cache = FFTCache(M, grid_size_x, grid_size_y, c̃, T)

    K̂ⱼ = Matrix{Complex{T}}(undef, N, M)
    L̂ⱼ = similar(K̂ⱼ)

    Φ_eval = Matrix{Complex{T}}(undef, N, M)

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
            # Φ_eval[i, j] = iszero(r) ? zero(Complex{T}) : Φ(abs.(pt), r, k, Yε_x1_cache, Yε_x2_cache)
            Φ_eval[i, j] = iszero(r) ? zero(Complex{T}) : Φ(r, k, Yε_cache)
        end
    end
    Φ_eval .*= exp.(-im * α .* x_grid)

    # Preallocate frequency-domain matrices
    Φ̂_freq = Matrix{Complex{T}}(undef, N, M)
    Ĝ_x1_freq = grad ? similar(Φ̂_freq) : nothing
    Ĝ_x2_freq = grad ? similar(Φ̂_freq) : nothing
    Ĥ_x1x1_freq = hess ? similar(Φ̂_freq) : nothing
    Ĥ_x1x2_freq = hess ? similar(Φ̂_freq) : nothing
    Ĥ_x2x2_freq = hess ? similar(Φ̂_freq) : nothing

    ## Transform to frequency domain with proper normalization
    # Shift spatial samples to FFT convention
    Φ̂_freq .= (2 * √(π * c̃)) / (N * M) .* fftshift(fft(fftshift(Φ_eval)))
    if grad || hess
        g_eval_sing_x1 .*= exp.(-im * α .* x_grid) .* x_grid
        g_eval_sing_x2 .*= exp.(-im * α .* x_grid) * y_grid'
        Ĝ_x1_freq .= (2 * √(π * c̃)) / (N * M) .* fftshift(fft(fftshift(g_eval_sing_x1)))
        Ĝ_x2_freq .= (2 * √(π * c̃)) / (N * M) .* fftshift(fft(fftshift(g_eval_sing_x2)))
        if hess
            h_eval_sing_x1x1 .*= exp.(-im * α .* x_grid) .* (x_grid .^ 2)
            h_eval_sing_x1x2 .*= exp.(-im * α .* x_grid) .* (x_grid * y_grid')
            h_eval_sing_x2x2 .*= exp.(-im * α .* x_grid) .* (y_grid' .^ 2)
            Ĥ_x1x1_freq .= (2 * √(π * c̃)) / (N * M) .* fftshift(fft(fftshift(h_eval_sing_x1x1)))
            Ĥ_x1x2_freq .= (2 * √(π * c̃)) / (N * M) .* fftshift(fft(fftshift(h_eval_sing_x1x2)))
            Ĥ_x2x2_freq .= (2 * √(π * c̃)) / (N * M) .* fftshift(fft(fftshift(h_eval_sing_x2x2)))
        end
    end

    # Precompute FFT plan (reused for each column)
    fft_plan = plan_fft!(fft_cache.shift_sample_eval_int)

    # Process each frequency component
    if hess
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, M, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            αₙ = α + j₁
            βₙ = abs(αₙ) <= k ? Complex{T}(√(k^2 - αₙ^2)) : im * √(αₙ^2 - k^2)

            # Compute L̂ⱼ, L̂ⱼ₁, L̂ⱼ₂, L̂ⱼ₁₁, L̂ⱼ₁₂, L̂ⱼ₂₂ coefficients
            @inbounds @batch for j ∈ 1:M
                j₂ = fft_cache.j2_idx[j]

                if j₂ * π / c̃ - βₙ == 0 || j₂ * π / c̃ + βₙ == 0
                    error("Division by zero encountered in frequency component computation for (i=$i, j=$j). Perturb parameters c̃.")
                end

                cst = (α + j₁)^2 + j₂^2 * π^2 / c̃^2 - k^2
                F̂ⱼ = -1 / cst * (-1 / (2 * √(π * c̃)) + im / 4 * Φ̂_freq[i, j])

                L̂ⱼ[i, j] = K̂ⱼ[i, j] - F̂ⱼ

                Ĝⱼ₁ = im * (α + j₁) * F̂ⱼ - Ĝ_x1_freq[i, j]
                Ĝⱼ₂ = im * j₂ * (π / c̃) * F̂ⱼ - Ĝ_x2_freq[i, j]
                L̂ⱼ₁[i, j] = im * (α + j₁) * K̂ⱼ[i, j] - Ĝⱼ₁
                L̂ⱼ₂[i, j] = im * j₂ * (π / c̃) * K̂ⱼ[i, j] - Ĝⱼ₂

                L̂ⱼ₁₁[i, j] = -(α + j₁)^2 * K̂ⱼ[i, j] - im * (α + j₁) * Ĝⱼ₁ + Ĥ_x1x1_freq[i, j]
                L̂ⱼ₁₂[i, j] = -(α + j₁) * j₂ * (π / c̃) * K̂ⱼ[i, j] - im * j₂ * (π / c̃) * Ĝⱼ₁ + Ĥ_x1x2_freq[i, j]
                L̂ⱼ₂₂[i, j] = -(j₂ * π / c̃)^2 * K̂ⱼ[i, j] - im * j₂ * (π / c̃) * Ĝⱼ₂ + Ĥ_x2x2_freq[i, j]
            end
        end
    elseif grad && !hess
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, M, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            αₙ = α + j₁
            βₙ = abs(αₙ) <= k ? Complex{T}(√(k^2 - αₙ^2)) : im * √(αₙ^2 - k^2)

            # Compute L̂ⱼ, L̂ⱼ₁, L̂ⱼ₂ coefficients
            @inbounds @batch for j ∈ 1:M
                j₂ = fft_cache.j2_idx[j]

                if j₂ * π / c̃ - βₙ == 0 || j₂ * π / c̃ + βₙ == 0
                    error("Division by zero encountered in frequency component computation for (i=$i, j=$j). Perturb parameters c̃.")
                end

                cst = (α + j₁)^2 + j₂^2 * π^2 / c̃^2 - k^2
                F̂ⱼ = -1 / cst * (-1 / (2 * √(π * c̃)) + im / 4 * Φ̂_freq[i, j])

                L̂ⱼ[i, j] = K̂ⱼ[i, j] - F̂ⱼ

                L̂ⱼ₁[i, j] = im * (α + j₁) * K̂ⱼ[i, j] - im * (α + j₁) * F̂ⱼ + Ĝ_x1_freq[i, j]
                L̂ⱼ₂[i, j] = im * j₂ * (π / c̃) * K̂ⱼ[i, j] - im * j₂ * (π / c̃) * F̂ⱼ + Ĝ_x2_freq[i, j]
            end
        end
    else
        @inbounds for i ∈ 1:N
            j₁, freq_idx, use_conj = process_frequency_component!(i, M, params, fft_cache, χ_cache, fft_plan, K̂ⱼ)

            αₙ = α + j₁
            βₙ = abs(αₙ) <= k ? Complex{T}(√(k^2 - αₙ^2)) : im * √(αₙ^2 - k^2)

            # Compute L̂ⱼ coefficients
            @inbounds @batch for j ∈ 1:M
                j₂ = fft_cache.j2_idx[j]

                if j₂ * π / c̃ - βₙ == 0 || j₂ * π / c̃ + βₙ == 0
                    error("Division by zero encountered in frequency component computation for (i=$i, j=$j). Perturb parameters c̃.")
                end

                cst = (α + j₁)^2 + j₂^2 * π^2 / c̃^2 - k^2
                F̂ⱼ = -1 / cst * (-1 / (2 * √(π * c̃)) + im / 4 * Φ̂_freq[i, j])

                L̂ⱼ[i, j] = K̂ⱼ[i, j] - F̂ⱼ
            end
        end
    end

    Lhat = copy(L̂ⱼ)

    # Transform back to spatial domain
    L_spatial = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ)))

    # Create spline interpolator
    value_interpolator = cubic_spline_interpolation((x_grid, y_grid), L_spatial; extrapolation_bc=Periodic())

    if grad && hess
        Lₙ₁ = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ₁)))
        Lₙ₂ = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ₂)))

        grad_interpolator = (∂x=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁; extrapolation_bc=Periodic()),
                             ∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₂; extrapolation_bc=Periodic()))

        Lₙ₁₁ = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ₁₁)))
        Lₙ₁₂ = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ₁₂)))
        Lₙ₂₂ = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ₂₂)))

        hess_interpolator = (∂x∂x=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁₁; extrapolation_bc=Periodic()),
                             ∂x∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁₂; extrapolation_bc=Periodic()),
                             ∂y∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₂₂; extrapolation_bc=Periodic()))

        return (value=value_interpolator,
                grad=grad_interpolator,
                hess=hess_interpolator,
                cache=Yε_cache)
    elseif grad && !hess
        Lₙ₁ = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ₁)))
        Lₙ₂ = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ₂)))

        grad_interpolator = (∂x=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁; extrapolation_bc=Periodic()),
                             ∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₂; extrapolation_bc=Periodic()))
        return (value=value_interpolator,
                grad=grad_interpolator,
                cache=Yε_cache)
    elseif !grad && hess
        Lₙ₁₁ = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ₁₁)))
        Lₙ₁₂ = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ₁₂)))
        Lₙ₂₂ = N * M / (2 * √(π * c̃)) .* fftshift(ifft!(fftshift(L̂ⱼ₂₂)))

        hess_interpolator = (∂x∂x=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁₁; extrapolation_bc=Periodic()),
                             ∂x∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₁₂; extrapolation_bc=Periodic()),
                             ∂y∂y=cubic_spline_interpolation((x_grid, y_grid), Lₙ₂₂; extrapolation_bc=Periodic()))
        return (value=value_interpolator,
                hess=hess_interpolator,
                cache=Yε_cache)
    end

    L_FourierSeries = FourierSeries(L̂ⱼ ./ (2 * √(π * c̃)); period=(2π, 2 * params.c_tilde),
                                    offset=(-grid_size_x - 1, -grid_size_y - 1))
    L_FourierSeries_alloc = FourierSeriesEvaluators.workspace_allocate(L_FourierSeries, (0.0, 0.0))


    ######## New BIE part and Non-uniform FFT - Type 2 ####
    R = 0.8
    start = -π + 0.2
    finish = π - 0.2
    xs = (start):(R - 0.15):(finish + R - 0.3)

    heights = (-0.4, 0.0, 0.4)
    list_centers = [(cx, cy) for cy ∈ heights for cx ∈ xs]

    # Set up the Single Layer potential Formulation
    N = params.N_bie

    fft_cache_BIE = params, L_FourierSeries_alloc, Yε_cache
    t, A = generate_SL_mat(params.k, N, R, 0.0, 0.0)
    A_inv = inv(A)

    # bd_pts_list = SVector{2 * N, Tuple{Float64, Float64}}[]
    # for (cx, cy) ∈ list_centers
    #     bd_pts = [curve_circle(ti, R; pos_x=cx, pos_y=cy) for ti ∈ t]
    #     push!(bd_pts_list, bd_pts)
    # end

    ####
    T = ComplexF64
    N_nufft = (2 * grid_size_x, 2 * grid_size_y)
    Np = 2 * N * length(list_centers)

    # pts_x = Float64[]
    # pts_y = Float64[]
    # real_pts_x = Float64[]
    # real_pts_y = Float64[]

    pts_x = Vector{Float64}(undef, Np)
    pts_y = Vector{Float64}(undef, Np)
    real_pts_x = Vector{Float64}(undef, Np)
    real_pts_y = Vector{Float64}(undef, Np)
    bd_pts_list = SVector{2 * N, Tuple{Float64, Float64}}[]

    # for bd_pts ∈ bd_pts_list
    #     for (x, y) ∈ bd_pts
    #         push!(pts_x, mod(x, 2π))
    #         push!(pts_y, mod((π / c̃) * y, 2π))
    #         push!(real_pts_x, x)
    #         push!(real_pts_y, y)
    #     end
    # end

    scale_y = (π / c̃)
    k = 1
    @inbounds for (cx, cy) ∈ list_centers
        for ti ∈ t
            x, y = curve_circle(ti, R; pos_x=cx, pos_y=cy)

            real_pts_x[k] = x
            real_pts_y[k] = y

            pts_x[k] = mod(x, 2π)
            pts_y[k] = mod(scale_y * y, 2π)

            k += 1
        end
        push!(bd_pts_list, [curve_circle(ti, R; pos_x=cx, pos_y=cy) for ti ∈ t])
    end

    plan_nufft = PlanNUFFT(T, N_nufft; m=HalfSupport(10))
    set_points!(plan_nufft, (pts_x, pts_y))
    wp = Array{T}(undef, Np)
    exec_type2!(wp, plan_nufft, fftshift(Lhat))
    ifft_nonuniform = wp ./ (2 * √(π * c̃))
    ####

    phi_list = Vector{SVector{2N, ComplexF64}}(undef, length(list_centers))
    prefac = (im / 4) * (π / N) * R
    @inbounds for idx ∈ eachindex(list_centers)
        cx, cy = list_centers[idx]

        off = (idx - 1) * 2 * N + 1
        b = generate_rhs_test(N, t, fft_cache_BIE, R, cx, cy, ifft_nonuniform, off)

        phi = A_inv * b
        phi_list[idx] = prefac .* phi
    end

    #######################################################

    return (value=value_interpolator,
            mat=L̂ⱼ,
            FourierSeries=L_FourierSeries_alloc,
            grid=(collect(x_grid), collect(y_grid)),
            cache=Yε_cache,
            bie_cache=(N=N, bd_pts=bd_pts_list, phi=phi_list, list_centers=list_centers))
end

"""
    eval_qp_green_bie(x, params::NamedTuple, cache_bie; nb_terms=40)

Evaluate the quasi-periodic Green's function at a point `x` using the FFT-BIE-based approach for points inside the domain `D_c` and the eigenfunction expansion for points outside.
"""
function eval_qp_green_bie(x, params::NamedTuple, cache_bie; nb_terms=40)

    α, k, c = (params.alpha, params.k, params.c)
    N, bd_pts_list, phi_list, list_centers = cache_bie

    # Check if the point is outside the domain D_c
    if abs(x[2]) > c
        return eigfunc_expansion(x, params; nb_terms=nb_terms)
    else
        t_period = get_t(x[1])

        x_norm = sqrt(t_period^2 + x[2]^2)

        idx_circle = find_closest_center(t_period, x[2], list_centers)
        bd_pts = bd_pts_list[idx_circle]
        phi = phi_list[idx_circle]

        return evaluate_potential_BIE(k, N, t_period, x[2], phi, bd_pts) +
               0.25im * Bessels.hankelh1(0, k * x_norm)
    end
end
