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
        for i = 1:N_states
            for k = 1:N_segments
                u_mid = 1 / 2 * (u[k+1, :] + u[k, :])
                x_mid = 1 / 2 * (x[k+1, :] + x[k, :]) + 1 / 8 * (t[k+1] - t[k]) * (f(x[k, :], u[k, :], p, t[k]) - f(x[k+1, :], u[k+1, :], p, t[k+1]))
            
                r[k+(i-1)*N_segments] = 1 / 6 * (t[k+1] - t[k]) *
                                        (f(x[k, :], u[k, :], p, t[k])[i] + 4 * f(x_mid, u_mid, p, t[k] + 1 / 2 * (t[k+1] - t[k]))[i] +
                                         f(x[k+1, :], u[k+1, :], p, t[k+1])[i]) .- (x[k+1, i] - x[k, i])
            end
        end
    end
    return r
end