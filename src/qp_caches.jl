# Data structures for integration parameters, normalization, and FFT-related caches used in 
# numerical integration and Fourier transforms.

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
    Binomial coefficients for the cutoff function
    """
    binom_coeffs::Vector{T1}

    """
    Precomputed exponents for the cutoff function
    """
    precomp_exponents::Vector{T2}

    """
    Order of the cutoff function
    """
    order::T2
end

function IntegrationParameters(a, b, order)
    binom_coeffs = Vector{eltype(a)}(undef, order + 1)
    precomp_exponents = Vector{eltype(order)}(undef, order + 1)

    a_minus_b = a - b
    denominator = 2 * order + 1

    for k ∈ 0:order
        binom_coeffs[k + 1] = binomial(order, k) * a_minus_b^k / (denominator - k)
        precomp_exponents[k + 1] = denominator - k
    end

    IntegrationParameters(a,
                          b,
                          binom_coeffs,
                          precomp_exponents,
                          order)
end

Base.:-(x::IntegrationParameters) = IntegrationParameters(-x.b, -x.a, x.order)


"""
    analytical_integration(t, _int::IntegrationParameters)

Evaluate the analytical integral of the polynomial cutoff function.
"""
function int_polynomial_cutoff(x, _int::IntegrationParameters)

    sum = zero(x)
    x_minus_a = x - _int.a

    for k ∈ 0:(_int.order)
        sum += _int.binom_coeffs[k + 1] * x_minus_a^_int.precomp_exponents[k + 1]
    end
    return sum
end

polynomial_cutoff(x, _int::IntegrationParameters) = (x - _int.a)^_int.order * (x - _int.b)^_int.order

function polynomial_cutoff_derivative(x, _int::IntegrationParameters)
    _int.order * (x - _int.a)^(_int.order - 1) * (x - _int.b)^_int.order +
    _int.order * (x - _int.a)^_int.order * (x - _int.b)^(_int.order - 1)
end

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
    Type of cutoff function (:polynomial or :mollifier)
    """
    type_cutoff::Symbol

    """
    Parameters of integration
    """
    params::IntegrationParameters{T1, T2}
end

function IntegrationCache(poly::IntegrationParameters, type_cutoff::Symbol)
    # Use change of variables to avoid cancellation errors
    if type_cutoff == :mollifier
        return IntegrationCache(1.0, type_cutoff, poly)
    else
        if poly.order <= 8
            return IntegrationCache(1 / ((poly.b - poly.a)^(2 * poly.order + 1) * (factorial(poly.order))^2 /
                                     factorial(2 * poly.order + 1)), type_cutoff, poly)
        else
            return IntegrationCache(1 / quadgk(x_ -> polynomial_cutoff(x_, poly), poly.a, poly.b)[1], type_cutoff, poly)
        end
    end
end

function mollifier(x, params::IntegrationParameters, k)
    u = (x - params.a) / (params.b - params.a)

    return exp(k * exp(-1 / u) / (u - 1))
end

function mollifier_derivative(x, params::IntegrationParameters, k)
    u = (x - params.a) / (params.b - params.a)

    exponent_from_exponent = -1 / u
    exponent = k * exp(exponent_from_exponent) / (u - 1)
    numerator = k * exp(exponent) * exp(exponent_from_exponent) * (u^2 - u + 1)
    denominator = (params.b - params.a) * u^2 * (u - 1)^2
    return -numerator / denominator
end

function mollifier_second_derivative(x, params::IntegrationParameters, k)
    u = (x - params.a) / (params.b - params.a)

    exponent_from_exponent = -1 / u
    exponent = k * exp(exponent_from_exponent) / (u - 1)

    f_t = exp(exponent)

    first_term = f_t / (params.b - params.a)^2
    second_term = k^2 * exp(2 * exponent_from_exponent) * (u^2 - u + 1)^2 / (u^4 * (u - 1)^4)
    third_term = -k * exp(exponent_from_exponent) *
                 ((u^2 - u + 1) / (u^4 * (u - 1)^2) + (2 * u - 1) * (-u^2 + u - 2) / (u^3 * (u - 1)^3))

    return first_term * (second_term + third_term)
end


struct FFTCache{T1 <: Real, T2 <: Real, T3 <: Integer}
    """
    Index of grid points -grid_size ≤ j1 ≤ grid_size-1
    """
    j1_idx::Vector{T3}

    """
    Index of grid points -grid_size ≤ j2 ≤ grid_size-1
    """
    j2_idx::Vector{T3}

    """
    Points to evaluate the fourier integral via 1D FFT
    """
    t_j_fft::Vector{T2}

    """
    Evaluation of the integrand (unshifted)
    """
    eval_int_fft_1D::Vector{Complex{T1}}

    """
    Shifted evaluations of the integrand
    """
    shift_sample_eval_int::Vector{Complex{T1}}

    """
    Evaluation of the Fourier integral
    """
    fft_eval::Vector{Complex{T1}}

    """
    Shifted evaluation of the Fourier integral
    """
    shift_fft_1d::Vector{Complex{T1}}

    """
    Flipped evaluation of the Fourier integral
    """
    fft_eval_flipped::Transpose{Complex{T1}, Vector{Complex{T1}}}
end


"""
    FFTCache(N::Integers, grid_size::Integer, params::NamedTuple, T=Float64)

Construct a containers for FFT operations.

# Arguments

  - `N`: total number of grid points in one dimension.
  - `grid_size`: Number of half of the grid points in one dimension.
  - `params`: Physical and numerical constants
  - `T`: Floating-point type for allocations. Defaults to `Float64`.

# Returns

An `FFTCache` object containing:

  - `j_idx`: Vector of integers for the computations of Fourier coefficients `[-grid_size, grid_size-1]`.

  - `t_j_fft`: Spatial grid points in `[-c̃, c̃]`.
  - Preallocated complex vectors for FFT operations:

      + `eval_int_fft_1D`: 1D integration using FFT.
      + `shift_sample_eval_int`: Shifted samples for FFT.
      + `fft_eval`: FFT evaluation.
      + `shift_fft_1d`: Shifted FFT result.
      + `fft_eval_flipped`: Transposed result of FFT.
"""
function FFTCache(M::Integer, grid_size_x::Integer, grid_size_y::Integer, c̃, ::Type{T}=Float64) where {T <: Real}

    j1_idx = Vector{Int}((-grid_size_x):(grid_size_x - 1))
    j2_idx = Vector{Int}((-grid_size_y):(grid_size_y - 1))
    t_j_fft = range(-c̃, c̃; length=2 * M + 1) |> collect

    # Preallocate all vectors with type `Complex{T}`
    eval_int_fft_1D = Vector{Complex{T}}(undef, 2 * M + 1)
    shift_sample_eval_int = Vector{Complex{T}}(undef, 2 * M)
    fft_eval = Vector{Complex{T}}(undef, 2 * M)
    shift_fft_1d = Vector{Complex{T}}(undef, 2 * M)
    fft_eval_flipped = transpose(Vector{Complex{T}}(undef, 2 * M))

    return FFTCache(j1_idx, j2_idx, t_j_fft, eval_int_fft_1D, shift_sample_eval_int, fft_eval, shift_fft_1d, fft_eval_flipped)
end

hankelh1(n, x::AbstractFloat) = Bessels.hankelh1(n, x)
hankelh1(n, x) = SpecialFunctions.hankelh1(n, x)
