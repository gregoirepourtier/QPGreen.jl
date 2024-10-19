# FFT-based algorithm to compute quasi-periodic Green's function for 2D Helmholtz equation

abstract type AbstractIntegrationCache end

struct IntegrationParameters{T1 <: Real, T2 <: Signed}
    a::T1
    b::T1
    order::T2
end

# invertBounds(x::IntegrationParameters) = IntegrationParameters(-x.b, -x.a, x.order)
Base.:-(x::IntegrationParameters) = IntegrationParameters(-x.b, -x.a, x.order)

polynomial_cutoff(x, _int::IntegrationParameters) = (x - _int.a)^_int.order * (x - _int.b)^_int.order
function polynomial_cutoff_derivative(x, _int::IntegrationParameters)
    _int.order * (x - _int.a)^(_int.order - 1) * (x - _int.b)^_int.order +
    _int.order * (x - _int.a)^_int.order * (x - _int.b)^(_int.order - 1)
end

int_polynomial_cutoff(x, _int::IntegrationParameters) = quadgk(x -> polynomial_cutoff(x, _int), _int.a, x)[1]

struct IntegrationCache{T1 <: Real, T2 <: Signed} <: AbstractIntegrationCache
    normalization::T1
    params::IntegrationParameters{T1, T2}
end

function IntegrationCache(poly::IntegrationParameters)
    IntegrationCache(1 / quadgk(x -> polynomial_cutoff(x, poly), poly.a, poly.b)[1], poly)
end

integrand_fourier_fft_1D(x, βⱼ, cache) = exp(im * βⱼ * x) * χ_der(x, cache)

"""
    fm_method_preparation(csts; grid_size=100, ε=0.4341)

Preparation step of the FFT-based algorithm.
Input arguments:

  - csts: tuple of constants (α, c, c̃, k, order)

Keyword arguments:

  - grid_size: size of the grid
  - ε: parameter of the cut-off function Yε

Returns the Fourier coefficients of the function Lₙ.
"""
function fm_method_preparation(csts; grid_size=100, ε=0.4341)

    α, c, c̃, k, order = csts
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
    j_idx = (-grid_size):(grid_size - 1)

    # Points to evaluate the fourier integral by 1D FFT
    t_j_fft = range(-c̃, c̃; length=(2 * N) + 1)

    # Allocate memory for the Fourier coefficients K̂ⱼ, F̂ⱼ, L̂ⱼ
    eval_int_fft_1D = Vector{Complex{typeof(α)}}(undef, 2 * N + 1)
    K̂ⱼ = Matrix{Complex{typeof(α)}}(undef, N, N)
    L̂ⱼ = Matrix{Complex{typeof(α)}}(undef, N, N)

    evaluation_Φ₁ = map(x -> norm(x) != 0 ? Φ₁(norm(x), cache_Yε) : 0.0, Iterators.product(xx, yy))
    # evaluation_Φ₁ = map(x -> Φ₁(norm(x), cache_Yε), Iterators.product(xx, yy))
    evaluation_Φ₂ = xx .* evaluation_Φ₁

    Φ̂₁ⱼ = (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_Φ₁)))
    Φ̂₂ⱼ = (2 * √(π * c̃)) / (N^2) .* fftshift(fft(fftshift(evaluation_Φ₂)))

    for i ∈ 1:N
        # a) Calculate Fourier Coefficients K̂ⱼ (using FFT to compute Fourier integrals)
        @views get_K̂ⱼ!(K̂ⱼ[i, :], eval_int_fft_1D, t_j_fft, j_idx, c̃, α, k, N, i, cache_χ)

        for j ∈ 1:N
            # b) Calculate Fourier Coefficients L̂ⱼ
            F̂₁ⱼ, F̂₂ⱼ = get_F̂ⱼ(j_idx[i], j_idx[j], c̃, Φ̂₁ⱼ[i, j], Φ̂₂ⱼ[i, j], cache_Yε)
            L̂ⱼ[j, i] = K̂ⱼ[i, j] - F̂₁ⱼ + im * α * F̂₂ⱼ
        end
    end

    Lₙ = N^2 / (2 * √(π * c̃)) .* fftshift(ifft(fftshift(L̂ⱼ)))

    return Lₙ
end


"""
    fm_method_calculation(x, csts, Lₙ, Yε; nb_terms=100)

Calculation step of the FFT-based algorithm.
Input arguments:

  - x: given as a 2D array
  - csts: tuple of the constants (α, c, c̃, k)
  - Lₙ: values of Lₙ at the grid points
  - Yε: function to build the cut-off function Yε

Keyword arguments:

      - nb_terms: number of terms in the series expansion

Returns the approximate value of the Green's function G(x).
"""
function fm_method_calculation(x, csts, Lₙ; nb_terms=100, ε=0.4341, order=8)

    α, c, c̃, _ = csts
    N, M = size(Lₙ)

    params_Yε = IntegrationParameters(ε, 2 * ε, order)
    cache_Yε = IntegrationCache(params_Yε)

    @assert N==M "Problem dimensions"

    # 2. Calculation
    if abs(x[2]) > c
        @info "The point is outside the domain D_c"
        return eigfunc_expansion(x, k, α; nb_terms=nb_terms)
    else
        @info "The point is inside the domain D_c"
        t = get_t(x[1])

        # Bicubic Interpolation to get Lₙ(t, x₂)
        xs = range(-π, π - π / (N / 2); length=N)
        ys = range(-c̃, c̃ - c̃ / (N / 2); length=N)
        interp_cubic = cubic_spline_interpolation((xs, ys), transpose(Lₙ)) # this can be move to the preparation step
        Lₙ_t_x₂ = interp_cubic(t, x[2])

        # Get K(t, x₂)
        K_t_x₂ = Lₙ_t_x₂ + f₁((t, x[2]), cache_Yε) - im * α * f₂((t, x[2]), cache_Yε)

        # Calculate the approximate value of G(x)
        G_x = exp(im * α * x[1]) * K_t_x₂

        return G_x
    end
end
