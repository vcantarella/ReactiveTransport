# I want to test different ODE formulations for the equilibrium sorption model.
using OrdinaryDiffEq
using NonlinearSolve
using CairoMakie
using SparseConnectivityTracer
using ADTypes

"""
    f_eq(c1, c2, smax, K1, K2)
Calculate the equilibrium sorbed concentrations for two components
     based on their concentrations and equilibrium constants using the Langmuir competitive sorption model.
"""
function f_eq(c1, c2 ,smax, K1, K2)
    s1_eq = smax * c1 / (K1*(1+c2/K2) + c1)
    s2_eq = smax * c2 / (K2*(1+c1/K1) + c2)
    return s1_eq, s2_eq
end

"""
    non_linear_function(u, p)
Calculate the error between the total concentration using current concentrations
to calculate the correct concentration that would be in equilibrium with the sorption phase.
"""
function nonlinear_function(u, p)
    c1 = u[1]
    c2 = u[2]
    c_tot1, c_tot2, smax, K1, K2, ϕ, ρₛ = p
    s1_eq, s2_eq = f_eq(c1, c2, smax, K1, K2)
    return [c_tot1 - c1*ϕ - s1_eq*ρₛ*(1-ϕ), c_tot2 - c2*ϕ - s2_eq*ρₛ*(1-ϕ)]
end
nonlinear_problem = NonlinearProblem(nonlinear_function, [0.1, 0.1, 0.1, 0.1],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# CSTR model for equilibrium sorption assuming we need to equilibrate the sorb and liquid concentrations
"""
    rhs_eq_sorption!(du, u, p, t)
Calculate the right-hand side of the ODE for the equilibrium sorption model.
Assuming we need to equilibrate the sorb and liquid concentrations to calculate the rates of exchange.
"""
function rhs_eq_sorption!(du, u, p, t)
    V, Q, c_in1, c_in2, smax, K1, K2, λ, ρₛ, ϕ = p
    c1 = u[1]
    c2 = u[2]
    s1 = u[3]
    s2 = u[4]
    c_c = u[5] # Conservative transport concentration
    c_tot1 = c1*ϕ + s1*ρₛ*(1-ϕ)
    c_tot2 = c2*ϕ + s2*ρₛ*(1-ϕ)
    nonlinear_problem = NonlinearProblem(nonlinear_function,
     [c1, c2],[c_tot1, c_tot2, smax, K1, K2, ϕ, ρₛ])
    sol = NonlinearSolve.solve(nonlinear_problem)
    # if sol.retcode != ReturnCode.Success
    #     error("Failed internal solution")
    # end
    c_eq = sol.u
    s_eq = f_eq(c_eq[1], c_eq[2], smax, K1, K2)
    du[1] = (Q/(V*ϕ)) * (c_in1 - c1) - (1-ϕ)/ϕ * ρₛ * λ * (s_eq[1] - s1)
    du[2] = (Q/(V*ϕ)) * (c_in2 - c2) - (1-ϕ)/ϕ * ρₛ * λ * (s_eq[2] - s2)
    du[3] = λ * (s_eq[1] - s1)
    du[4] = λ * (s_eq[2] - s2)
    du[5] = (Q/(V*ϕ)) * (c_in1 - c_c)
end

# CSTR model for equilibrium sorption assuming we need to equilibrate the sorb and liquid concentrations
"""
    rhs_eq_sorption_v2!(du, u, p, t)
Calculate the right-hand side of the ODE for the equilibrium sorption model.
Assuming we DON'T need to equilibrate the sorb and liquid concentrations to calculate the rates of exchange.
"""
function rhs_eq_sorption_v2!(du, u, p, t)
    V, Q, c_in1, c_in2, smax, K1, K2, λ, ρₛ, ϕ = p
    c1 = u[1]
    c2 = u[2]
    s1 = u[3]
    s2 = u[4]
    c_c = u[5]
    s_eq = f_eq(c1, c2, smax, K1, K2)
    du[1] = (Q/(V*ϕ)) * (c_in1 - c1) - (1-ϕ)/ϕ * ρₛ * λ * (s_eq[1] - s1)
    du[2] = (Q/(V*ϕ)) * (c_in2 - c2) - (1-ϕ)/ϕ * ρₛ * λ * (s_eq[2] - s2)
    du[3] = λ * (s_eq[1] - s1)
    du[4] = λ * (s_eq[2] - s2)
    du[5] = (Q/(V*ϕ)) * (c_in1 - c_c)
end

u0 = [0, 0, 0, 0, 0] # Initial concentrations
c_in1 = 2e-3 # Inlet concentration
c_in2 = 1e-3 # Inlet concentration
V = 0.1 # l Volume of the reactor
Q = 2e-3 # Flow rate in l/s
smax = 0.0034 # Maximum sorbed concentration
K1 = 1e-3 # Equilibrium constant
K2 = 2e-4 # Equilibrium constant
λ = 0.1e-2 # Rate constant for sorption
ρₛ = 2.65 # Density of the sorbent [kg/l]
ϕ = 0.3 # Porosity
p = [V, Q, c_in1, c_in2, smax, K1, K2, λ, ρₛ, ϕ]
du0 = zeros(eltype(u0), length(u0))
# detector = TracerSparsityDetector()
# jac_sparsity2 = ADTypes.jacobian_sparsity((du, u) -> rhs_eq_sorption!(du, u, p, 1),
#     du0, u0, detector) # add the sparsity pattern to speed up the solution

tspan = (0.0, 10000.0) # Time span for the simulation
# Shut the system down with a callback
condition(u, t, integrator) = t == 5000.0
affect!(integrator) = integrator.p[3] = 0.0 # Set c_in to 0 at t = 5000s
cb = DiscreteCallback(condition, affect!)
cb2 = DiscreteCallback(condition, affect!)
prob = ODEProblem(rhs_eq_sorption!, u0, tspan, p)
sol = solve(prob, Rosenbrock23(), abstol=1e-8, reltol=1e-8, callback=cb, tstops = [5000.0])
p = [V, Q, c_in1, c_in2, smax, K1, K2, λ, ρₛ, ϕ]
prob_v2 = ODEProblem(rhs_eq_sorption_v2!, u0, tspan, p)
sol_v2 = solve(prob_v2, Tsit5(), abstol=1e-8, reltol=1e-8,callback=cb, tstops = [5000.0])

## Plotting the results
### --------------------------------------------------
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time (s)",
    ylabel="Concentration (mol L⁻¹ or kg⁻¹)",
    width=600, height=400)
hlines!(ax, smax, label="Smax", color=:black)
lines!(ax, sol.t, [u[1] for u in sol.u], label="C1 - s_eq_iter", color=:blue)
lines!(ax, sol.t, [u[2] for u in sol.u], label="C2 - s_eq_iter", color=:lightblue)
lines!(ax, sol.t, [u[3] for u in sol.u], label="S1 - s_eq_iter", color=:red)
lines!(ax, sol.t, [u[4] for u in sol.u], label="S2 - s_eq_iter", color=:darkred)
lines!(ax, sol.t, [u[3] + u[4] for u in sol.u], label="S1 + S2 - s_eq_iter", color=:crimson)
lines!(ax, sol.t, [u[5] for u in sol.u], label="Cconservative", color=:orange, linestyle=:dash)
#lines!(ax, sol_v2.t, [u[5] for u in sol_v2.u], label="C_c (v2)", color=:orange, linestyle=:dash)
lines!(ax, sol_v2.t, [u[1] for u in sol_v2.u], label="C1 - ode", color=:blue, linestyle=:dash)
lines!(ax, sol_v2.t, [u[2] for u in sol_v2.u], label="C2 - ode", color=:lightblue, linestyle=:dash)
lines!(ax, sol_v2.t, [u[3] for u in sol_v2.u], label="S1 - ode", color=:red, linestyle=:dash)
lines!(ax, sol_v2.t, [u[4] for u in sol_v2.u], label="S2 - ode", color=:darkred, linestyle=:dash)
lines!(ax, sol.t, [f_eq(c_in1, c_in2, smax, K1, K2)[1] for _ in sol.t], label="Seq (1)", color=:green, linestyle=:dash)
lines!(ax, sol.t, [f_eq(c_in1, c_in2, smax, K1, K2)[2] for _ in sol.t], label="Seq (2)", color=:darkgreen, linestyle=:dash)
lines!(ax, sol.t, [u[3] + u[4] for u in sol.u], label="S1 + S2 - ode", color=:crimson, linestyle=:dash)
Legend(fig[1,2],ax, position=:rc)
resize_to_layout!(fig)
fig
save("cstr_sorption_results.png", fig)
## Checking the mass balance
### --------------------------------------------------
# For a CSTR with sorption, mass balance should be:
# d(total_mass)/dt = mass_in - mass_out

# Total mass in the reactor at any time
function calculate_total_mass(u, ϕ, ρₛ, V)
    c1, c2, s1, s2, c_c = u
    # Total mass = liquid mass + sorbed mass
    total_mass_1 = c1 * ϕ * V + s1 * ρₛ * (1-ϕ) * V
    total_mass_2 = c2 * ϕ * V + s2 * ρₛ * (1-ϕ) * V
    return total_mass_1, total_mass_2
end

# Mass flow rates
function calculate_mass_flows(u, p, t)
    V, Q, c_in1, c_in2, smax, K1, K2, λ, ρₛ, ϕ = p
    c1, c2, s1, s2, c_c = u
    # Mass in (only through liquid flow)
    mass_in_1 = Q * c_in1
    mass_in_2 = Q * c_in2
    # Mass out (only through liquid flow)
    mass_out_1 = Q * c1
    mass_out_2 = Q * c2
    # Net mass flow
    net_flow_1 = mass_in_1 - mass_out_1
    net_flow_2 = mass_in_2 - mass_out_2
    
    return net_flow_1, net_flow_2
end

# Calculate the "estimated" equilibrium sorption for each time step
for i in [1:5]
    u = sol.u[i]
    c1, c2, s1, s2, c_c = u
    c_tot1 = c1 * ϕ + s1 * ρₛ * (1-ϕ)
    c_tot2 = c2 * ϕ + s2 * ρₛ * (1-ϕ)
    p_nonlinear = [c_tot1, c_tot2, smax, K1, K2, ϕ, ρₛ]
    nonlinear_problem = NonlinearProblem(nonlinear_function,
     [c1, c2],p_nonlinear)
    sol_nonlinear = NonlinearSolve.solve(nonlinear_problem, abstol=1e-8, reltol=1e-8)
    @show sol_nonlinear
    c_eq = sol_nonlinear.u
    s_eq = f_eq(c_eq[1], c_eq[2], smax, K1, K2)
    @show s_eq
    @show c_eq
    #@show c1, c2, s1, s2
end
