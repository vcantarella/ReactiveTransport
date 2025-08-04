# I want to test different ODE formulations for the equilibrium sorption model.
using OrdinaryDiffEq
using NonlinearSolve
using CairoMakie
using SparseConnectivityTracer
using ADTypes


function f_eq(c,smax, K)
    s_eq = smax * c / (K + c)
    return s_eq
end
function non_linear_function(u, p)
    c = u[1]
    c_tot, smax, K, ϕ, ρₛ = p
    s_eq = f_eq(c, smax, K)
    return [c_tot - c*ϕ - s_eq*ρₛ*(1-ϕ)]
end
nonlinear_problem = NonlinearProblem(non_linear_function, [0.1, 0.1], [0.0, 0.0, 0.0, 0.0, 0.0])
# CSTR model for equilibrium sorption
function rhs_eq_sorption!(du, u, p, t)
    V, Q, c_in, smax, K, λ, ρₛ, ϕ = p
    c = u[1]
    s = u[2]
    c_c = u[3] # Conservative transport concentration
    c_tot = c*ϕ + s*ρₛ*(1-ϕ)
    remake(nonlinear_problem, u = [c], p = [c_tot, smax, K, ϕ, ρₛ])
    sol = NonlinearSolve.solve(nonlinear_problem, NonlinearSolve.NewtonRaphson(), abstol=1e-8, reltol=1e-8)
    c_eq = sol.u
    s_eq = f_eq(c_eq[1], smax, K)
    du[1] = (Q/V) * (c_in - c) - (1-ϕ)/ϕ * ρₛ * λ * (s_eq - s)
    du[2] = λ * (s_eq - s)
    du[3] = (Q/V) * (c_in - c_c)
end

function rhs_eq_sorption_v2!(du, u, p, t)
    V, Q, c_in, smax, K, λ, ρₛ, ϕ = p
    c = u[1]
    s = u[2]
    c_c = u[3]
    s_eq = f_eq(c, smax, K)
    du[1] = (Q/V) * (c_in - c) - (1-ϕ)/ϕ * ρₛ * λ * (s_eq - s)
    du[2] = λ * (s_eq - s)
    du[3] = (Q/V) * (c_in - c_c)
end

u0 = [0, 0, 0] # Initial concentrations
c_in = 2e-3 # Inlet concentration
V = 0.1 # Volume of the reactor
Q = 2e-3 # Flow rate in l/s
smax = 0.0034 # Maximum sorbed concentration
K = 1e-2 # Equilibrium constant
λ = 0.1e-2 # Rate constant for sorption
ρₛ = 2.65 # Density of the sorbent [kg/l]
ϕ = 0.3 # Porosity
p = [V, Q, c_in, smax, K, λ, ρₛ, ϕ]
du0 = zeros(eltype(u0), length(u0))
detector = TracerSparsityDetector()
jac_sparsity2 = ADTypes.jacobian_sparsity((du, u) -> rhs_eq_sorption!(du, u, p, 1),
    du0, u0, detector) # add the sparsity pattern to speed up the solution

tspan = (0.0, 10000.0) # Time span for the simulation
# Shut the system down with a callback
condition(u, t, integrator) = t == 5000.0
affect!(integrator) = integrator.p[3] = 0.0 # Set c_in to 0 at t = 5000s
cb = DiscreteCallback(condition, affect!)
cb2 = DiscreteCallback(condition, affect!)
prob = ODEProblem(rhs_eq_sorption!, u0, tspan, p)
sol = solve(prob, Rosenbrock23(), abstol=1e-8, reltol=1e-8, callback=cb, tstops = [5000.0])
p = [V, Q, c_in, smax, K, λ, ρₛ, ϕ]
prob_v2 = ODEProblem(rhs_eq_sorption_v2!, u0, tspan, p)
sol_v2 = solve(prob_v2, Tsit5(), abstol=1e-12, reltol=1e-12, callback=cb, tstops = [5000.0])

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Concentration (mol L⁻¹ or kg⁻¹)")
lines!(ax, sol.t, [u[3] for u in sol.u], label="C_c", color=:orange)
lines!(ax, sol.t, [u[1] for u in sol.u], label="C", color=:blue)
lines!(ax, sol.t, [u[2] for u in sol.u], label="S", color=:red)
lines!(ax, sol_v2.t, [u[3] for u in sol_v2.u], label="C_c (v2)", color=:orange, linestyle=:dash)
lines!(ax, sol_v2.t, [u[1] for u in sol_v2.u], label="C (v2)", color=:blue, linestyle=:dash)
lines!(ax, sol_v2.t, [u[2] for u in sol_v2.u], label="S (v2)", color=:red, linestyle=:dash)
lines!(ax, sol.t, [f_eq(c_in, smax, K) for _ in sol.t], label="Seq ", color=:green, linestyle=:dash)
axislegend(ax, position=:rc)
fig