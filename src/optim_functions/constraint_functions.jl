function cons!(r, optim_var, p_optim)
    # Unpack decision variables
    u = reshape(optim_var[1:p_optim[:size_u][1]*p_optim[:size_u][2]], p_optim[:size_u])
    x = reshape(optim_var[p_optim[:size_u][1]*p_optim[:size_u][2].+(1:p_optim[:size_x][1]*p_optim[:size_x][2])], p_optim[:size_x])
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

function get_cons_jac_prototype(optim_cons_lb, p_optim, optim_var_guess)
    function cons_oop(x)
        r = zeros(eltype(x), length(optim_cons_lb))
        cons!(r, x, p_optim)
        return r
    end

    J_dense = jacobian(cons_oop, AutoForwardDiff(), rand(length(optim_var_guess)))
    cons_jac_prototype = sparse(J_dense)

    return cons_jac_prototype
end