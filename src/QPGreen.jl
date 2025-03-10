module QPGreen

using LinearAlgebra
using FFTW
using QuadGK
using Interpolations

using DocStringExtensions
using Polyester
using StaticArrays

import SpecialFunctions
import Bessels

using .MathConstants: eulergamma

include("expansions.jl")
export image_expansion, eigfunc_expansion, eigfunc_expansion_derivative, image_expansion_derivative

include("fft_caches.jl")

include("other_methods/lattice_sums.jl")
include("other_methods/ewald.jl")
include("fft_eval.jl")
export fm_method_preparation, fm_method_calculation
export fm_method_calculation_smooth

# include("other_methods/api.jl")

include("cutoff_functions.jl")
include("fft_helpers.jl")

include("fft_gradient.jl")
export fm_method_preparation_derivative, fm_method_calculation_derivative
export fm_method_calculation_derivative_smooth



end # module