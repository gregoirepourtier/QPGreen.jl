# FFT-based algorithm to compute quasi-periodic Green's function for 2D Helmholtz equation

abstract type AbstractIntegrationCache end

"""
$(TYPEDEF)

Structure storing the parameters of integration for the cutoff functions.

$(TYPEDFIELDS)
"""
struct IntegrationParameters{T1 <: Real, T2 <: Signed}
    """
    Lower bound
    """
    a::T1

    """
    Upper bound
    """
    b::T1

    """
    Order of the cutoff function
    """
    order::T2
end

Base.:-(x::IntegrationParameters) = IntegrationParameters(-x.b, -x.a, x.order)

polynomial_cutoff(x, _int::IntegrationParameters) = (x - _int.a)^_int.order * (x - _int.b)^_int.order
function polynomial_cutoff_derivative(x, _int::IntegrationParameters)
    _int.order * (x - _int.a)^(_int.order - 1) * (x - _int.b)^_int.order +
    _int.order * (x - _int.a)^_int.order * (x - _int.b)^(_int.order - 1)
end
int_polynomial_cutoff(x, _int::IntegrationParameters) = quadgk(x_ -> polynomial_cutoff(x_, _int), _int.a, x)[1]

"""
$(TYPEDEF)

Structure storing the normalization factor and the parameters of integration for the cutoff functions.

$(TYPEDFIELDS)
"""
struct IntegrationCache{T1 <: Real, T2 <: Signed} <: AbstractIntegrationCache
    """
    Normalization factor
    """
    normalization::T1

    """
    Parameters of integration
    """
    params::IntegrationParameters{T1, T2}
end

function IntegrationCache(poly::IntegrationParameters)
    IntegrationCache(1 / quadgk(x_ -> polynomial_cutoff(x_, poly), poly.a, poly.b)[1], poly)
end

integrand_fourier_fft_1D(x, βⱼ, cache) = exp(im * βⱼ * x) * χ_der(x, cache)

"""
    fm_method_preparation(csts, grid_size)

Preparation step of the FFT-based algorithm.
Input arguments:

  - csts: tuple of constants (α, k, c, c̃, ε, order) (recommended value: 0.4341)
  - grid_size: size of the grid

Returns the Fourier coefficients of the function Lₙ.
"""
function fm_method_preparation(csts::NamedTuple, grid_size::Integer)

    α, k, c, c̃, ε, order = (csts.α, csts.k, csts.c, csts.c̃, csts.ε, csts.order)
    c₁, c₂ = (c, (c + c̃) / 2)

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

    # Allocate memory for the Fourier coefficients K̂ⱼ, F̂ⱼ, L̂ⱼ
    type_α = typeof(α)

    eval_int_fft_1D = Vector{Complex{type_α}}(undef, 2 * N + 1)
    shift_sample_eval_int = Vector{Complex{type_α}}(undef, 2 * N)
    fft_eval = Vector{Complex{type_α}}(undef, 2 * N)
    shift_fft_1d = Vector{Complex{type_α}}(undef, 2 * N)
    fft_eval_flipped = transpose(Vector{Complex{type_α}}(undef, 2 * N))

    K̂ⱼ = Matrix{Complex{type_α}}(undef, N, N)
    L̂ⱼ = Matrix{Complex{type_α}}(undef, N, N)

    evaluation_Φ₁ = Matrix{type_α}(undef, N, N)
    evaluation_Φ₂ = Matrix{type_α}(undef, N, N)

    for i ∈ eachindex(xx)
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

    p = plan_fft!(shift_sample_eval_int; flags=FFTW.ESTIMATE, timelimit=Inf)

    for i ∈ 1:N
        # a) Calculate Fourier Coefficients K̂ⱼ (using FFT to compute Fourier integrals)
        @views get_K̂ⱼ!(K̂ⱼ[i, :], eval_int_fft_1D, fft_eval_flipped, t_j_fft, j_idx, c̃, α, k, N, i, cache_χ, p,
                        shift_sample_eval_int, fft_eval, shift_fft_1d)
        j₁ = j_idx[i]

        # b) Calculate Fourier Coefficients L̂ⱼ
        if i > N ÷ 2 + 1
            idx_fft_row = N - i + 2
            for j ∈ 1:N
                j₂ = j_idx[j]
                F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ[idx_fft_row, j], Φ̂₂ⱼ[idx_fft_row, j], cache_Yε, type_α)
                L̂ⱼ[j, i] = K̂ⱼ[i, j] - F̂₁ⱼ + im * α * F̂₂ⱼ
            end
        else
            idx_fft_row = i
            for j ∈ 1:N
                j₂ = j_idx[j]
                F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j₁, j₂, c̃, Φ̂₁ⱼ[idx_fft_row, j], conj(Φ̂₂ⱼ[idx_fft_row, j]), cache_Yε, type_α)
                L̂ⱼ[j, i] = K̂ⱼ[i, j] - F̂₁ⱼ + im * α * F̂₂ⱼ
            end
        end
    end

    Lₙ = N^2 / (2 * √(π * c̃)) .* fftshift(ifft(fftshift(L̂ⱼ)))

    # Create the Bicubic Interpolation function
    interp_cubic = cubic_spline_interpolation((xx, yy), transpose(Lₙ))
    # interp_cubic = interpolate((xx, yy), transpose(Lₙ), Gridded(Linear()))    

    return Lₙ, interp_cubic, cache_Yε
end


"""
    fm_method_calculation(x, csts, Lₙ, interp_cubic; nb_terms=100)

Calculation step of the FFT-based algorithm.
Input arguments:

  - x: given as a 2D array
  - csts: tuple of the constants (α, k, c, c̃, ε, order)
  - Lₙ: values of Lₙ at the grid points
  - interp_cubic: bicubic interpolation function
  - cache_Yε: cache for the cut-off function Yε

Keyword arguments:

      - nb_terms: number of terms in the series expansion

Returns the approximate value of the Green's function G(x).
"""
function fm_method_calculation(x, csts::NamedTuple, Lₙ, interp_cubic::T, cache_Yε::IntegrationCache; nb_terms=100) where {T}

    α, k, c = (csts.α, csts.k, csts.c)
    N, M = size(Lₙ)

    @assert N==M "Problem dimensions"

    if abs(x[2]) > c
        return eigfunc_expansion(x, k, α; nb_terms=nb_terms)
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
