
function create_orig_model(dx)
    function rhs!(du, u, p, t)
        v = p[1]
        De = p[2]
        c_in = p[3]
        # transport
        c_advec = [[c_in];u]
        advec = -v .* diff(c_advec) ./ dx
        gradc = diff(u)./dx
        disp = ([gradc; [zero(eltype(u))]]-[[zero(eltype(u))]; gradc]).* De ./ dx
        du[:] .= advec .+ disp
    end
    return rhs!
end

function create_non_alloc_ADEST(dx)
    function non_alloc_ADEST!(du, u, p ,t)
        n_rows = size(u, 1)
        v = p[1]
        De = p[2]
        c_in = p[3]
        
        # Calculate transport terms directly without temporary arrays
        # First cell (boundary condition)
        du[1] = -v * (u[1] - c_in) / dx
        
        # Calculate dispersion at first cell
        grad_fwd = (u[2] - u[1]) / dx
        du[1] += De * (grad_fwd - zero(eltype(u))) / dx
        
        # Interior cells
        for i in 2:n_rows-1
            # Advection
            du[i] = -v * (u[i] - u[i-1]) / dx
            
            # Dispersion
            grad_fwd = (u[i+1] - u[i]) / dx
            grad_bwd = (u[i] - u[i-1]) / dx
            du[i] += De * (grad_fwd - grad_bwd) / dx
        end
        
        # Last cell
        du[n_rows] = -v * (u[n_rows] - u[n_rows-1]) / dx
        grad_bwd = (u[n_rows] - u[n_rows-1]) / dx
        du[n_rows] += De * (zero(eltype(u)) - grad_bwd) / dx  # Zero-gradient at boundary
    end
    return non_alloc_ADEST!
end


# Normally we want conservative transport model that finds the porosity and dispersivitive based on a tracer experiment

function create_tracer_rhs(D, c_in, dx, q)
    function tracer_primitive!(du, u, p, t)
        n_rows = size(u, 1)
        ϕ = p[1]
        αₗ = p[2]
        # q_l = Float64(q.(t))
        v = q / ϕ  # Velocity based on flow rate and porosity
        De = D + αₗ * v  # Dispersion coefficient
        
        # Calculate transport terms directly without temporary arrays
        # First cell (boundary condition)
        du[1] = -v * (u[1] - c_in) / dx
        
        # Calculate dispersion at first cell
        grad_fwd = (u[2] - u[1]) / dx
        du[1] += De * (grad_fwd) / dx
        
        # Interior cells
        for i in 2:n_rows-1
            # Advection
            du[i] = -v * (u[i] - u[i-1]) / dx
            
            # Dispersion
            grad_fwd = (u[i+1] - u[i]) / dx
            grad_bwd = (u[i] - u[i-1]) / dx
            du[i] += De * (grad_fwd - grad_bwd) / dx
        end
        
        # Last cell
        du[n_rows] = -v * (u[n_rows] - u[n_rows-1]) / dx
        grad_bwd = (u[n_rows] - u[n_rows-1]) / dx
        du[n_rows] += De * (zero(eltype(u)) - grad_bwd) / dx  # Zero-gradient at boundary
    end
    return tracer_primitive!
end


function create_tracer_rhs_q_callback(D, c_in, dx)
    function tracer_primitive!(du, u, p, t)
        n_rows = size(u, 1)
        ϕ = p[1]
        αₗ = p[2]
        q = p[3]  # Flow rate as a function of time
        # q_l = Float64(q.(t))
        v = q / ϕ  # Velocity based on flow rate and porosity
        De = D + αₗ * v  # Dispersion coefficient
        
        # Calculate transport terms directly without temporary arrays
        # First cell (boundary condition)
        du[1] = -v * (u[1] - c_in) / dx
        
        # Calculate dispersion at first cell
        grad_fwd = (u[2] - u[1]) / dx
        du[1] += De * (grad_fwd - zero(eltype(u))) / dx
        
        # Interior cells
        for i in 2:n_rows-1
            # Advection
            du[i] = -v * (u[i] - u[i-1]) / dx
            
            # Dispersion
            grad_fwd = (u[i+1] - u[i]) / dx
            grad_bwd = (u[i] - u[i-1]) / dx
            du[i] += De * (grad_fwd - grad_bwd) / dx
        end
        
        # Last cell
        du[n_rows] = -v * (u[n_rows] - u[n_rows-1]) / dx
        grad_bwd = (u[n_rows] - u[n_rows-1]) / dx
        du[n_rows] += De * (zero(eltype(u)) - grad_bwd) / dx  # Zero-gradient at boundary
    end
    return tracer_primitive!
end

function create_tracer_rhs_q_fun(D, c_in, dx, q)
    function tracer_primitive!(du, u, p, t)
        n_rows = size(u, 1)
        ϕ = p[1]
        αₗ = p[2]
        q_l = @inline Float64(q(t))
        v = q_l / ϕ  # Velocity based on flow rate and porosity
        De = D + αₗ * v  # Dispersion coefficient
        
        # Calculate transport terms directly without temporary arrays
        # First cell (boundary condition)
        du[1] = -v * (u[1] - c_in) / dx
        
        # Calculate dispersion at first cell
        grad_fwd = (u[2] - u[1]) / dx
        du[1] += De * grad_fwd / dx
        
        # Interior cells
        for i in 2:n_rows-1
            # Advection
            du[i] = -v * (u[i] - u[i-1]) / dx
            
            # Dispersion
            grad_fwd = (u[i+1] - u[i]) / dx
            grad_bwd = (u[i] - u[i-1]) / dx
            du[i] += De * (grad_fwd - grad_bwd) / dx
        end
        
        # Last cell
        du[n_rows] = -v * (u[n_rows] - u[n_rows-1]) / dx
        grad_bwd = (u[n_rows] - u[n_rows-1]) / dx
        du[n_rows] -= De * grad_bwd / dx  # Zero-gradient at boundary
    end
    return tracer_primitive!
end