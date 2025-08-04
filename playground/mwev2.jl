using OrdinaryDiffEq
using Plots
function rhs(du, u, p, t)
    V, Q, c_in = p
    c = u[1]
    du[1] = (Q/V) * (c_in - c)
end
tspan = (0.0, 10000.0) # Time span for the simulation
u0 = [0.0] # Initial concentrations
c_in = 2e-3 # Inlet concentration
V = 0.1 # Volume of the reactor
Q = 2e-3 # Flow rate in l/s
p = [V, Q, c_in]
@show p
condition(u, t, integrator) = t == 5000.0
affect!(integrator) = integrator.p[3] = 0.0 # Set c_in to 0 at t = 5000s
cb = DiscreteCallback(condition, affect!)

alias = ODEAliasSpecifier(alias_p = false)
prob = ODEProblem(rhs, u0, tspan, p)
sol = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8,
           callback=cb, tstops = [5000.0], alias = alias)

@show p  # Original p should be unchanged
fig = plot(sol, xlabel="Time (s)", ylabel="Concentration (mol/l)", title="CSTR Concentration Dynamics")
display(fig)