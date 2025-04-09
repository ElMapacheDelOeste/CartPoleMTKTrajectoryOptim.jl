using CartPoleMTKTrajectoryOptim
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as Dt
using ModelingToolkitStandardLibrary
using Multibody
using OrdinaryDiffEq
using Optimization, OptimizationMOI, Ipopt
using DataInterpolations
import GLMakie

@mtkmodel CartPolePartial begin
    @components begin
        world = World(n=[0.0, 0.0, -1.0])
        rail = Prismatic(n=[1, 0, 0], axisflange=true)
        cart = Body(m=1.0)
        rev = Revolute(n=[0, 1, 0])
        pole = Body(m=0.3, r_cm=[0.0, 0.0, -0.5])
        actuator = Force()
    end
    @equations begin
        connect(world.frame_b, rail.frame_a)
        connect(rail.frame_b, cart.frame_a)
        connect(actuator.frame_a, rail.frame_a)
        connect(actuator.frame_b, rail.frame_b)
        connect(cart.frame_a, rev.frame_a)
        connect(rev.frame_b, pole.frame_a)
    end
end
@mtkmodel CartPoleOptim begin
    @extend CartPolePartial()
    @equations begin
        actuator.force.u ~ [0.0, 0.0, 0.0]
    end
end

@named mdl = CartPoleOptim()
mdl = complete(mdl)
multi_ = multibody(mdl)
ssys = structural_simplify(multi_)
odeprob = ODEProblem(ssys, [], (0.0, 1.0))

# Generate control function to be used in the trajectory optimization - xdot = f(x,u,p,t)
(; f, dvs, ps, io_sys) = ModelingToolkit.generate_control_function(multi_, [mdl.actuator.force.u[1]]);
f_oop(x, u, p, t) = f(similar(x), x, u, p, t) # f_oop : (x,u,p,t)->xdot(t)

## Optimization function parameters ##
p_optim = (N_segments=25,
           method="Hermite-Simpson",
           dyn_func=f_oop,
           N_states=length(odeprob.u0),
           mdl_params=odeprob.p)

## Decision variables : bounds and initial guesses ##
# optim_var[1] : timeseries of force.u(t)
# optim_var[2] : timeseries of states(t)
# optim_var[3] : t_final

# Control input variables
u_t0 = Float64[]
u_tf = [0.0]
u_guess = zeros(p_optim[:N_segments] + 1)
lb_u = [-20.0]
ub_u = [20.0]
lb_u_vec = vec(collect(reduce(hcat, [lb_u for i = 1:p_optim[:N_segments]+1]))')
ub_u_vec = vec(collect(reduce(hcat, [ub_u for i = 1:p_optim[:N_segments]+1]))')
# State variables
x_t0 = [0.0, 0.0, 0.0, 0.0] # [phi, s, omega, v]
x_tf = [-pi, 1.0, 0.0, 0.0] # [phi, s, omega, v]
x_guess = collect(reduce(hcat, ([x_t0 + (x_tf - x_t0) * (i - 1) / p_optim[:N_segments] for i = 1:p_optim[:N_segments]+1]))')
lb_x = [-2*pi, 0.0, -Inf, -Inf] # [phi, s, omega, v]
ub_x = [pi, 3.0, Inf, Inf] # [phi, s, omega, v]
lb_x_vec = vec(collect(reduce(hcat, [lb_x for i = 1:p_optim[:N_segments]+1]))')
ub_x_vec = vec(collect(reduce(hcat, [ub_x for i = 1:p_optim[:N_segments]+1]))')
# Duration of the trajectory
t_final_guess = 2.0
lb_t_final = 2.0
ub_t_final = 2.0

optim_var_guess = vcat(vec(u_guess), vec(x_guess), t_final_guess)
optim_var_lb = vcat(lb_u_vec, lb_x_vec, lb_t_final)
optim_var_ub = vcat(ub_u_vec, ub_x_vec, ub_t_final)

## Add parameters for the optimization functions ##
p_optim = (p_optim...,
           x_t0=x_t0,
           x_tf=x_tf,
           u_t0=u_t0,
           u_tf=u_tf,
           size_u=(size(u_guess, 1), size(u_guess, 2)),
           size_x=(size(x_guess, 1), size(x_guess, 2)),
           size_t_final=(size(t_final_guess, 1), size(t_final_guess, 2)))

## Constraints bounds ##
optim_cons_lb = zeros(length(vcat(u_t0, u_tf, x_t0, x_tf)) + p_optim[:N_segments] * p_optim[:N_states])
optim_cons_ub = zeros(length(vcat(u_t0, u_tf, x_t0, x_tf)) + p_optim[:N_segments] * p_optim[:N_states])

cons_jac_prototype = get_cons_jac_prototype(optim_cons_lb, p_optim, optim_var_guess)

## Optimization - Problem construction ##
using SparseConnectivityTracer: TracerSparsityDetector, TracerLocalSparsityDetector
opt_fun = OptimizationFunction(least_effort, AutoSparse(AutoForwardDiff(); sparsity_detector = TracerLocalSparsityDetector());
                               cons=cons!)

opt_prob = OptimizationProblem(opt_fun, optim_var_guess, p_optim;
                               lcons=optim_cons_lb,
                               ucons=optim_cons_ub,
                               lb=optim_var_lb,
                               ub=optim_var_ub)

opt_solver = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
                                                         "hessian_approximation" => "limited-memory",
                                                         "print_level" => 5,
                                                         "max_iter" => 100)

opt_solve = solve(opt_prob, opt_solver)

### Reshape solution ###
f_sol = reshape(opt_solve.u[1:p_optim[:size_u][1]*p_optim[:size_u][2]], p_optim[:size_u])
x_sol = reshape(opt_solve.u[(size(f_sol, 1)*size(f_sol, 2)).+(1:size(x_guess)[1]*size(x_guess)[2])], size(x_guess))
t_final_sol = opt_solve.u[end]

### Test solution through simulation ###
tspan_sol = (0.0, t_final_sol)

f_force = DataInterpolations.LinearInterpolation(f_sol[:,1], collect(0.0:t_final_sol/p_optim[:N_segments]:t_final_sol), extrapolate=true)

@mtkmodel CartPoleSimTest begin
    @extend CartPolePartial()
    @components begin
        force_input = ModelingToolkitStandardLibrary.Blocks.TimeVaryingFunction(f=f_force)
    end
    @equations begin
        actuator.force.u .~ [force_input.output.u, 0.0, 0.0]
    end
end

@named mdl_test_sol = CartPoleSimTest()
mdl_test_sol = complete(mdl_test_sol)
multi_test_sol = multibody(mdl_test_sol)
ssys_test_sol = structural_simplify(multi_test_sol)

odeprob_test_sol = ODEProblem(ssys_test_sol, x_sol[1, :], (0.0, t_final_sol))
odeprob_test_sol.u0 = x_sol[1, :]
sol_test_sol = solve(odeprob_test_sol, Rodas4(), saveat=(t_final_sol / p_optim[:N_segments]))

fig, _ = Multibody.render(mdl_test_sol, sol_test_sol, 0.0, up=[0, 0, 1], x=0, y=7, z=0, lookat=[0, 0, 0])
fig
### END Simulation ###
