function cons!(r, optim_var, p_optim)
    # Unpack decision variables
    # t_final = popat!(optim_var, length(optim_var))
    optim_var_mat = collect(reshape(optim_var[1:end-1],p_optim[:size_x][2]+ p_optim[:size_u][2], p_optim[:size_u][1])')
    u = optim_var_mat[:, 1:p_optim[:size_u][2]]
    x = optim_var_mat[:, p_optim[:size_u][2]+1:end]
    t_final = optim_var[end]
    
    # Unpack parameters
    N = p_optim[:N_segments]

    # Boundary conditions
    idx_bc = 0
    for n in eachindex(p_optim[:u_t0])
        r[idx_bc+n] = u[1, n] - p_optim[:u_t0][n]
    end
    idx_bc += length(p_optim[:u_t0])

    for n in eachindex(p_optim[:u_tf])
        r[idx_bc+n] = u[N+1, n] - p_optim[:u_tf][n]
    end
    idx_bc += length(p_optim[:u_tf])

    for n in eachindex(p_optim[:x_t0])
        r[idx_bc+n] = x[1, n] - p_optim[:x_t0][n]
    end
    idx_bc += length(p_optim[:x_t0])

    for n in eachindex(p_optim[:x_tf])
        r[idx_bc+n] = x[N+1, n] - p_optim[:x_tf][n]
    end
    idx_bc += length(p_optim[:x_tf])

    # Dynamics constraints -> Hermite-Simpson collocation - compressed form
    t = range(0, t_final, length=N + 1)
    r_dyn = dyn_contraints_direct_collocation(x, u, p_optim[:mdl_params], t, N, p_optim[:N_states]; f=p_optim[:dyn_func], method=p_optim[:method])
    # Pack dynamics constraints residuals
    r[idx_bc+1:end] .= r_dyn
end

function dyn_contraints_direct_collocation(x, u, p, t, N_segments, N_states; f=Nothing, method="Hermite-Simpson")
    # x : model state matrix of size (N_segments+1) x N_states
    # u : model control input matrix of size (N_segments+1) x N_controls
    # p : vector of model parameters
    # t : vector of time points, corresponding to the discretization mesh
    # N_segments : number of segments
    # N_states : number of states
    # f : dynamics function of the form x_dot = f(x, u, p, t)
    # method is the method to use for the collocation (Trapezoidal or Hermite-Simpson)

    type = promote_type(eltype(x), eltype(u), eltype(p), eltype(t))
    r = zeros(type, N_segments * N_states)
    if method == "Trapezoidal"
        for k = 1:N_segments
            for i = 1:N_states
                r[(N_states*(k-1)+i)] = 1 / 2 * (t[k+1] - t[k]) * (f(x[k, :], u[k, :], p, t[k])[i] + f(x[k+1, :], u[k+1, :], p, t[k+1])[i]) .-
                                        (x[k+1, i] - x[k, i])
            end
        end
    elseif method == "Hermite-Simpson"
        for k = 1:N_segments
            for i = 1:N_states
                u_mid = 1 / 2 * (u[k+1, :] + u[k, :])
                x_mid = 1 / 2 * (x[k+1, :] + x[k, :]) + 1 / 8 * (t[k+1] - t[k]) * (f(x[k, :], u[k, :], p, t[k]) - f(x[k+1, :], u[k+1, :], p, t[k+1]))
            
                r[(k-1)*N_states+i] = 1 / 6 * (t[k+1] - t[k]) *
                                        (f(x[k, :], u[k, :], p, t[k])[i] + 4 * f(x_mid, u_mid, p, t[k] + 1 / 2 * (t[k+1] - t[k]))[i] +
                                         f(x[k+1, :], u[k+1, :], p, t[k+1])[i]) .- (x[k+1, i] - x[k, i])
            end
        end
    end
    return r
end

function get_cons_jac_prototype(optim_cons_lb, p_optim, optim_var_lb)
    function cons_oop(x)
        r = zeros(eltype(x), length(optim_cons_lb))
        cons!(r, x, p_optim)
        return r
    end

    J_dense = jacobian(cons_oop, AutoForwardDiff(), rand(length(optim_var_lb)))
    J_ones_dense = Float64.((J_dense .!= 0))
    cons_jac_prototype = sparse(J_ones_dense)

    return cons_jac_prototype
end