function least_effort(optim_var, p_optim)
    # Unpack decision variables
    u = reshape(optim_var[1:p_optim[:size_u][1]*p_optim[:size_u][2]], p_optim[:size_u])
    t_final = optim_var[end]
    # Unpack parameters
    N_seg = p_optim[:N_segments]

    # Trapezoidal rule for integral approximation
    return sum(sum(1 / 2 * (t_final / N_seg) * (u[k, :] .^ 2 + u[k+1, :] .^ 2) for k = 1:N_seg))
end