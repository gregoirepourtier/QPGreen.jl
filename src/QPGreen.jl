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

include("lattice_sums.jl")
include("ewald.jl")
include("fft_method.jl")
export fm_method_preparation, fm_method_calculation

include("api.jl")

include("cutoff_functions.jl")
include("fm.jl")

include("gradient.jl")
export analytical_derivative, fm_method_preparation_derivative, fm_method_calculation_derivative


end # module