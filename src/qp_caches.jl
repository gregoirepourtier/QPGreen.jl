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


struct FFTCache{T1 <: Real, T2 <: Real, T3 <: Integer}
    """
    Index of grid points -grid_size ≤ j ≤ grid_size-1
    """
    j_idx::Vector{T3}

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
function FFTCache(N::Integer, grid_size::Integer, c̃, ::Type{T}=Float64) where {T <: Real}

    j_idx = Vector{Int}((-grid_size):(grid_size - 1))
    t_j_fft = range(-c̃, c̃; length=2 * N + 1) |> collect

    # Preallocate all vectors with type `Complex{T}`
    eval_int_fft_1D = Vector{Complex{T}}(undef, 2 * N + 1)
    shift_sample_eval_int = Vector{Complex{T}}(undef, 2 * N)
    fft_eval = Vector{Complex{T}}(undef, 2 * N)
    shift_fft_1d = Vector{Complex{T}}(undef, 2 * N)
    fft_eval_flipped = transpose(Vector{Complex{T}}(undef, 2 * N))

    return FFTCache(j_idx, t_j_fft, eval_int_fft_1D, shift_sample_eval_int, fft_eval, shift_fft_1d, fft_eval_flipped)
end
