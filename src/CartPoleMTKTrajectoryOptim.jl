module CartPoleMTKTrajectoryOptim

using DifferentiationInterface
using SparseArrays

include("optim_functions/constraint_functions.jl")
export cons!, get_cons_jac_prototype

include("optim_functions/objective_functions.jl")
export least_effort

end # module CartPoleMTKTrajectoryOptim
