# First Order derivatives of the Green's function

"""
    analytical_derivative(z, csts; period=2π, nb_terms=100)

Calculate the derivative of the Green's function using the eigenfunction expansion.
Input arguments:

  - z: 2D point at which the derivative is evaluated
  - csts: Named tuple of the constants for the problem definition

Keyword arguments:

  - period: period of the Green's function
  - nb_terms: number of terms in the series expansion

Returns the value of the derivative of the Green's function.
"""
function analytical_derivative(z, csts::NamedTuple; period=2π, nb_terms=100)

    G_prime_x1 = zero(Complex{eltype(z)})
    G_prime_x2 = zero(Complex{eltype(z)})

    range_term = nb_terms ÷ 2

    α, k = (csts.α, csts.k)

    # Compute the value of the derivative of the Green function by basic eigenfunction expansion
    for n ∈ (-range_term):range_term
        αₙ = α + n
        βₙ = abs(αₙ) <= k ? √(k^2 - αₙ^2) : im * √(αₙ^2 - k^2)

        exp_term = exp(im * αₙ * z[1] + im * βₙ * abs(z[2]))
        G_prime_x1 += im / (2 * period) * im * αₙ / βₙ * exp_term
        G_prime_x2 += im / (2 * period) * im * sign(z[2]) * exp_term
    end

    return G_prime_x1, G_prime_x2
end


"""
    get_Ĥⱼ(j₁, j₂, csts, F̂₂ⱼ, Ψ̂₁ⱼ₁, Ψ̂₁ⱼ₂, Ψ̂₁ⱼ₃, cache::IntegrationCache, ::Type{type_α})

Calculate the Fourier coefficients Ĥⱼ.
"""
function get_Ĥⱼ(j₁, j₂, csts::NamedTuple, F̂₂ⱼ, Ψ̂₁ⱼ₁, Ψ̂₁ⱼ₂, Ψ̂₁ⱼ₃, cache::IntegrationCache, ::Type{type_α}) where {type_α}
    c̃, k, α = (csts.c̃, csts.k, csts.α)
    if (j₁^2 + j₂^2) ≠ 0
        cst = 1 / (j₁^2 + j₂^2 * π^2 / c̃^2)
        Λ̂₁ⱼ = cst * (1 / (2 * π) * Ψ̂₁ⱼ₁ + im * j₁ / (4 * √(π * c̃)))
        Λ̂₂ⱼ = cst * (-2 * im * j₁ * Λ̂₁ⱼ + 1 / (2 * π) * Ψ̂₁ⱼ₂)
        Λ̂₃ⱼ = cst * (-2 * im * j₁ * Λ̂₂ⱼ + 1 / (2 * π) * Ψ̂₁ⱼ₃)
        Ĥ₁ⱼ = -k^2 / 2 * F̂₂ⱼ + Λ̂₁ⱼ - im * α * Λ̂₂ⱼ - α^2 / 2 * Λ̂₃ⱼ
        Ĥ₂ⱼ = 0
    else # special case |j| = 0
        integral = quadgk(x -> f₀_Ĥⱼ(x, cache), 0.0, cache.params.b)[1]

        Ĥ₁ⱼ = im * α / (4 * √(π * c̃)) * integral
        Ĥ₂ⱼ = zero(Complex{type_α})
    end

    return Ĥ₁ⱼ, Ĥ₂ⱼ
end

"""
fm_method_preparation_derivative(csts::NamedTuple, grid_size::Integer)

Preparation of the Fourier coefficients for the first order derivative of the Green's function.
"""
function fm_method_preparation_derivative(csts::NamedTuple, grid_size::Integer)
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

    # index of grid points -grid_size ≤ j ≤ grid_size-1
    j_idx = collect((-grid_size):(grid_size - 1))

    # Points to evaluate the fourier integral by 1D FFT
    t_j_fft = collect(range(-c̃, c̃; length=(2 * N) + 1))

    # Allocate memory for the Fourier coefficients K̂ⱼ, L̂ⱼ and the evaluation of the Fourier integrals
    eval_int_fft_1D = Vector{Complex{type_α}}(undef, 2 * N + 1)

    shift_sample_eval_int = Vector{Complex{type_α}}(undef, 2 * N)
    fft_eval = Vector{Complex{type_α}}(undef, 2 * N)
    shift_fft_1d = Vector{Complex{type_α}}(undef, 2 * N)

    fft_eval_flipped = transpose(Vector{Complex{type_α}}(undef, 2 * N))

    K̂ⱼ = Matrix{Complex{type_α}}(undef, N, N)
    L̂ⱼ = Matrix{Complex{type_α}}(undef, N, N)

    evaluation_Φ₁ = Matrix{type_α}(undef, N, N)
    evaluation_Φ₂ = Matrix{type_α}(undef, N, N)

    evaluation_Ψ₁₁ = Matrix{type_α}(undef, N, N)
    evaluation_Ψ₁₂ = Matrix{type_α}(undef, N, N)
    evaluation_Ψ₁₃ = Matrix{type_α}(undef, N, N)

    for i ∈ eachindex(xx)
        for j ∈ eachindex(yy)
            x = xx[i]
            y = yy[j]
            evaluation_Φ₁[i, j] = norm((x, y)) != 0.0 ? Φ₁(norm((x, y)), cache_Yε) : zero(type_α)
            evaluation_Ψ₁₁[i, j] = norm((x, y)) != 0.0 ? Ψ₁₁((x, y), norm((x, y)), cache_Yε) : zero(type_α)
        end
    end
    evaluation_Φ₂ .= xx .* evaluation_Φ₁

    evaluation_Ψ₁₂ .= xx .* evaluation_Ψ₁₁
    evaluation_Ψ₁₃ .= xx .* evaluation_Ψ₁₂

    Φ̂₁ⱼ = Matrix{Complex{type_α}}(undef, N, N)
    Φ̂₂ⱼ = Matrix{Complex{type_α}}(undef, N, N)

    Ψ̂₁ⱼ₁ = Matrix{Complex{type_α}}(undef, N, N)
    Ψ̂₁ⱼ₂ = Matrix{Complex{type_α}}(undef, N, N)
    Ψ̂₁ⱼ₃ = Matrix{Complex{type_α}}(undef, N, N)

    Φ̂₁ⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_Φ₁)))
    Φ̂₂ⱼ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_Φ₂)))

    Ψ̂₁ⱼ₁ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_Ψ₁₁)))
    Ψ̂₁ⱼ₂ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_Ψ₁₂)))
    Ψ̂₁ⱼ₃ .= (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_Ψ₁₃)))

    p = plan_fft!(shift_sample_eval_int)

    for i ∈ 1:N
        # a) Calculate Fourier Coefficients K̂ⱼ (using FFT to compute Fourier integrals)
        @views get_K̂ⱼ!(K̂ⱼ[:, i], t_j_fft, eval_int_fft_1D, shift_sample_eval_int, fft_eval, shift_fft_1d, fft_eval_flipped,
                        j_idx, c̃, α, k, N, i, cache_χ, p)
        j₁ = j_idx[i]

        for j ∈ 1:N
            j₂ = j_idx[j]
            # b) Calculate Fourier Coefficients L̂ⱼ
            F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ[i, j], Φ̂₂ⱼ[i, j], cache_Yε, type_α)
            Ĥ₁ⱼ, Ĥ₂ⱼ = get_Ĥⱼ(j₁, j₂, csts, F̂₂ⱼ, Ψ̂₁ⱼ₁[i, j], Ψ̂₁ⱼ₂[i, j], Ψ̂₁ⱼ₃[i, j], cache_Yε, type_α)

            L̂ⱼ[j, i] = im * (α + j₁) * K̂ⱼ[j, i] - Ĥ₁ⱼ
            # L̂ⱼ[j, i] = K̂ⱼ[j, i] - F̂₁ⱼ + im * α * F̂₂ⱼ
        end
    end

    Lₙ = N^2 / (2 * √(π * c̃)) .* transpose(fftshift(ifft!(fftshift(L̂ⱼ))))

    # Create the Bicubic Interpolation function
    interp_cubic = cubic_spline_interpolation((xx, yy), Lₙ)
    # interp_cubic = linear_interpolation((xx, yy), Lₙ; extrapolation_bc=Line()) # More efficient but less accurate

    return Lₙ, interp_cubic, cache_Yε
end

function fm_method_calculation_derivative(x, csts::NamedTuple, Lₙ, interp_cubic::T, cache_Yε::IntegrationCache;
                                          nb_terms=100) where {T}

    α, c = (csts.α, csts.c)
    N, M = size(Lₙ)

    @assert N==M "Problem dimensions"

    if abs(x[2]) > c
        return analytical_derivative(x, csts; nb_terms=nb_terms)
    else
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        Lₙ_t_x₂ = interp_cubic(t, x[2])

        # Get K(t, x₂)
        K₁_t_x₂ = Lₙ_t_x₂ + h₁((t, x[2]), csts, cache_Yε)
        # K₁_t_x₂ = Lₙ_t_x₂ + f₁((t, x[2]), cache_Yε) - im * α * f₂((t, x[2]), cache_Yε)
        K₂_t_x₂ = 0.0 # Lₙ_t_x₂ + h₂((t, x[2]), k, α, cache_Yε)

        # Calculate the approximate value of G(x)
        der_G_x₁ = exp(im * α * x[1]) * K₁_t_x₂
        der_G_x₂ = exp(im * α * x[1]) * K₂_t_x₂

        return (der_G_x₁, der_G_x₂)
    end
end
