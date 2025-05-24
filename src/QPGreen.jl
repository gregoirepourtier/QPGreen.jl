module QPGreen

using LinearAlgebra
using FFTW

import SpecialFunctions
import Bessels
using QuadGK
using Interpolations
using DocStringExtensions
using Polyester
using StaticArrays

using .MathConstants: eulergamma

include("expansions.jl")
export image_expansion, image_expansion_derivative, image_expansion_smooth, image_expansion_derivative_smooth
export eigfunc_expansion, eigfunc_expansion_derivative

include("qp_caches.jl")

include("other_methods/lattice_sums.jl")
include("other_methods/ewald.jl")
include("api.jl")
export init_qp_green_fft
export eval_qp_green, eval_smooth_qp_green
export grad_qp_green, grad_smooth_qp_green

include("cutoff_functions.jl")
include("qp_fft_helpers.jl")

end # module