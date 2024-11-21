# Helper functions

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


struct FFT_cache{T1 <: Integer, T2 <: Real, T3 <: Real}
    """
    index of grid points -grid_size ≤ j ≤ grid_size-1
    """
    j_idx::Vector{T1}

    """
    Points to evaluate the fourier integral by 1D FFT
    """
    t_j_fft::Vector{T2}

    """
    Evaluation of the Fourier integrals
    """
    eval_int_fft_1D::Vector{Complex{T3}}

    """
    """
    shift_sample_eval_int::Vector{Complex{T3}}

    """
    """
    fft_eval::Vector{Complex{T3}}

    """
    """
    shift_fft_1d::Vector{Complex{T3}}

    """
    """
    fft_eval_flipped::Transpose{Complex{T3}, Vector{Complex{T3}}}
end

function FFT_cache(N, grid_size::Integer, csts::NamedTuple, ::Type{type_α}) where {type_α}

    c̃ = csts.c̃

    j_idx = collect((-grid_size):(grid_size - 1))
    t_j_fft = collect(range(-c̃, c̃; length=(2 * N) + 1))

    eval_int_fft_1D = Vector{Complex{type_α}}(undef, 2 * N + 1)

    shift_sample_eval_int = Vector{Complex{type_α}}(undef, 2 * N)
    fft_eval = Vector{Complex{type_α}}(undef, 2 * N)
    shift_fft_1d = Vector{Complex{type_α}}(undef, 2 * N)

    fft_eval_flipped = transpose(Vector{Complex{type_α}}(undef, 2 * N))

    FFT_cache(j_idx, t_j_fft, eval_int_fft_1D, shift_sample_eval_int, fft_eval, shift_fft_1d, fft_eval_flipped)
end
