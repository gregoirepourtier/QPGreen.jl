module QPGreen

using LinearAlgebra
using FFTW
using QuadGK
using Interpolations

using DocStringExtensions

import SpecialFunctions
import Bessels

include("expansions.jl")
export image_expansion, eigfunc_expansion

include("utils.jl")

include("other_methods/lattice_sums.jl")
include("other_methods/ewald.jl")
include("fft_method.jl")
export fm_method_preparation, fm_method_calculation

include("other_methods/api.jl")

include("cutoff_functions.jl")
include("fm.jl")

include("gradient.jl")
export analytical_derivative, fm_method_preparation_derivative, fm_method_calculation_derivative



end # module