include("../src/analytical_functions.jl")
include("../src/ODE_conservative_transport.jl")

using Test
using OrdinaryDiffEq
using SciMLSensitivity
using FileIO
using StaticArrays

struct FlowRate{TQ<:AbstractVector, TTime<:AbstractVector}
    Q::TQ
    start_times::TTime
    end_times::TTime
end

function (f::FlowRate)(t)
    len = length(f.Q)
    @inbounds for i in eachindex(f.start_times)
        if t >= f.start_times[i] && t <= f.end_times[i]
            return f.Q[i]
        elseif i < len && t > f.end_times[i] && t < f.start_times[i+1]
            return (f.Q[i]+ f.Q[i+1])/ 2
        end
    end
    return zero(eltype(f.Q))
end

flow_rate_ds = load("test/test_data/br_flow_rate.jld2")
column = 1
#Q_f = FlowRate(SVector{length(Q_1)}(Q_1), SVector{length(Q_1)}(start_times), SVector{length(Q_1)}(end_times))

function create_ode_from_flow(D, c_in, dx, A, Q, start_times, end_times)
    Q_f = FlowRate(SVector{length(Q)}(Q), SVector{length(Q)}(start_times), SVector{length(Q)}(end_times))
    q_f = (t, A) -> Q_f(t)/A
    q(A) = (t) -> q_f(t, A)
    rhs! = create_tracer_rhs_q_fun(D, c_in, dx, q(A))
    return rhs!
end
# benchmark tests
using BenchmarkTools
function test_benchmark_ode_functions()
    # Define parameters
    dx = 0.0001 # Spatial step size
    L = 0.08 #m (8 cm)  # Spatial locations
    D = 3.5*1e-2 #cm to m
    A = π * D^2 / 4 # Cross-sectional area
    Q = 2 # ml/hr
    Q = Q*1e-6 / 3600 # Convert to m^3/s
    x = range(0, stop=L, step=dx)  # Spatial locations
    t = range(0.1, stop=72000, length=100)  # Time locations
    c0 = 1.0e-3
    c_in = 1e-3
    ϕ = 0.3
    v = Q / (A * ϕ)  # Velocity based on flow rate and porosity
    alpha_l = 8e-4
    Dl = 1e-9 + alpha_l * v
    t_pulse = 36000.0
    cr = zeros( length(t))
    cr_pulse = zeros(length(t))
    u0 = zeros(length(x))
    du0 = copy(u0)
    p_basic = [v, Dl, c_in]
    p_tracer = [ϕ, alpha_l]
    p_tracer_v2 = [ϕ, alpha_l, q_v2(A)(100_000)]
    display(@benchmark $q($A)($10_000))
    display(@benchmark $q_v2($A)($10_000))
    rhs_basic! = create_non_alloc_ADEST(dx)
    rhs_tracer! = create_tracer_rhs_q_callback(Dl, c_in, dx)
    Q_1 = flow_rate_ds["Q"][flow_rate_ds["column"] .== column]
    start_times = convert.(Float64, flow_rate_ds["start_times"][flow_rate_ds["column"] .== column])
    end_times = convert.(Float64, flow_rate_ds["end_times"][flow_rate_ds["column"] .== column])
    rhs_tracer_v2! = create_ode_from_flow(Dl, c_in, dx, A, Q_1, start_times, end_times)
    display(@benchmark $rhs_basic!($du0, $u0, $p_basic, $100_000))
    display(@benchmark $rhs_tracer!($du0, $u0, $p_tracer_v2, $100_000))
    display(@benchmark $rhs_tracer_v2!($du0, $u0, $p_tracer, 100_000))
end

test_benchmark_ode_functions()

function test_type_stability()
    # Define parameters
    dx = 0.0001 # Spatial step size
    L = 0.08 #m (8 cm)  # Spatial locations
    D = 3.5*1e-2 #cm to m
    A = π * D^2 / 4 # Cross-sectional area
    Q = 2 # ml/hr
    Q = Q*1e-6 / 3600 # Convert to m^3/s
    x = range(0, stop=L, step=dx)  # Spatial locations
    t = range(0.1, stop=72000, length=100)  # Time locations
    c0 = 1.0e-3
    c_in = 0.0
    ϕ = 0.3
    v = Q / (A * ϕ)  # Velocity based on flow rate and porosity
    alpha_l = 8e-4
    Dl = 1e-9 + alpha_l * v
    t_pulse = 36000.0
    cr = zeros( length(t))
    cr_pulse = zeros(length(t))
    u0 = zeros(length(x))
    du0 = copy(u0)
    p_basic = [v, Dl, c_in]
    p_tracer = [ϕ, alpha_l]
    Q_1 = flow_rate_ds["Q"][flow_rate_ds["column"] .== column]
    start_times = convert.(Float64, flow_rate_ds["start_times"][flow_rate_ds["column"] .== column])
    end_times = convert.(Float64, flow_rate_ds["end_times"][flow_rate_ds["column"] .== column])
    rhs_tracer_v2! = create_ode_from_flow(Dl, c_in, dx, A, Q_1, start_times, end_times)

    rhs_basic! = create_non_alloc_ADEST(dx)
    rhs_tracer! = create_tracer_rhs(Dl, c_in, dx, q(A))
    display(@code_warntype rhs_basic!(du0, u0, p_basic, 100_000))
    display(@code_warntype rhs_tracer!(du0, u0, p_tracer, 100_000))
    display(@code_warntype rhs_tracer_v2!(du0, u0, p_tracer, 100_000))
end
test_type_stability()


## Test that constant injection outputs the same result as pulse injection when t < t_pulse
@testset "ODE Functions" begin
    # Define parameters
    dx = 0.0001 # Spatial step size
    L = 0.08 #m (8 cm)  # Spatial locations
    D = 3.5*1e-2 #cm to m
    A = π * D^2 / 4 # Cross-sectional area
    Q = 2 # ml/hr
    Q = Q*1e-6 / 3600 # Convert to m^3/s
    x = range(0+dx/2, stop=L-dx/2, step=dx)  # Spatial locations
    t = range(0.1, stop=72000, length=100)  # Time locations
    c0 = 0.0
    c_in = 1e-3
    ϕ = 0.3
    v = Q / (A * ϕ)  # Velocity based on flow rate and porosity
    alpha_l = 8e-4
    Dl = 1e-9 + alpha_l * v
    
    cr_pulse = zeros(length(t))
    u0 = zeros(length(x))
    du0 = copy(u0)
    p_basic = [v, Dl, c_in]
    p_tracer = [ϕ, alpha_l]
    # Calculate concentration profiles
    f! = create_orig_model(dx)
    rhs! = create_non_alloc_ADEST(dx)
    du1 = zeros(eltype(u0), length(u0))
    du2 = zeros(eltype(u0), length(u0))
    f!(du1, u0, p_basic, t[1])
    rhs!(du2, u0, p_basic, t[1])
    du1 == du2 # Check that the two functions produce the same result for the first time step
    prob = ODEProblem(f!, u0, (t[1], t[end]), p_basic)
    sol = solve(prob, Tsit5(), saveat=t, abstol=1e-16, reltol=1e-12)
    u_ = sol.u[40]
    f!(du1, u_, p_basic, t[1])
    rhs!(du2, u_, p_basic, t[1])
    # check in which index there is a difference
    differences = findall(@. !(isapprox.(du1, du2, atol=1e-8)))

    du1 == du2
    prob_new = ODEProblem(rhs!, u0, (t[1], t[end]), p_basic)
    sol_new = solve(prob_new, Tsit5(), saveat=t, abstol=1e-16, reltol=1e-12)
    @test sol_new.u == sol.u
    # Check that the results are the same for t < t_pulse
    cr = zeros(length(t))
    constant_injection!(cr, sol.t, L, c_in, c0, v, Dl)
    result = [sol.u[i][end] for i in eachindex(sol.t)]
    new_result = [sol_new.u[i][end] for i in eachindex(sol_new.t)]
    # plot
    using CairoMakie
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Concentration (mol/L)")
    lines!(ax, sol.t, result, label="ODE - vectorized", color=:blue)
    lines!(ax, sol_new.t, new_result, label="ODE - loop", linestyle=:dash)
    lines!(ax, sol.t, cr, label="Analytical function", linestyle=:dash)
    axislegend(ax, position=:lt)
    fig
    # Check that the results are the same for t < t_pulse
    @test all(isapprox.(result, cr, atol=1e-3))
    @test all(isapprox.(new_result, cr, atol=1e-3))
end

# Test differentiability of the analytical functions
using ForwardDiff
@testset "Differentiability" begin
   # Define parameters
    dx = 0.0001 # Spatial step size
    L = 0.08 #m (8 cm)  # Spatial locations
    D = 3.5*1e-2 #cm to m
    A = π * D^2 / 4 # Cross-sectional area
    Q = 2 # ml/hr
    Q = Q*1e-6 / 3600 # Convert to m^3/s
    x = range(0, stop=L, step=dx)  # Spatial locations
    t = range(0.1, stop=72000, length=100)  # Time locations
    c0 = 0.0
    c_in = 1e-3
    ϕ = 0.3
    v = Q / (A * ϕ)  # Velocity based on flow rate and porosity
    alpha_l = 8e-4
    Dl = 1e-9 + alpha_l * v
    u0 = zeros(length(x))
    du0 = copy(u0)
    p_basic = [v, Dl, c_in]
    p_tracer = [ϕ, alpha_l]
    # Calculate concentration profiles
    f! = create_non_alloc_ADEST(dx)

    prob = ODEProblem(f!, u0, (t[1], t[end]), p_basic)
    sol = solve(prob, QNDF(), saveat=t, abstol=1e-8, reltol=1e-8)

    function ddu_du(u, p, t)
        du = zeros(eltype(u), length(u))
        f!(du, u, p, t)
        return du
    end
    # Test the Jacobian of the ODE function
    jacobian_result = ForwardDiff.jacobian(u -> ddu_du(u, p_basic, 0), u0)
    @test size(jacobian_result) == (length(u0), length(u0))
end
