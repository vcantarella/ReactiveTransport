using SpecialFunctions
using ArgCheck
"""
constant_injection(cr, x, t, c0, c_in, v, Dl)

Calculates the concentration profile of a solute undergoing
advection-dispersion transport in a porous 1D domain with constant
 in a 1D domain with constant injection
Original reference: (Ogata & Banks, 1961): 
Appelo, C.A.J.; Postma, Dieke. Geochemistry, Groundwater and Pollution (p. 104).

# Arguments
- `cr::Matrix`: A 2D array to store the concentration profile of a solute. 
    first dimension is time and second dimension is space.
- `x::Real`: location. (Where to sample the concentration)
- `t::Vector`: A 1D array of time locations. (When to sample the concentration)
- `c0::Real`: The concentratio at x=0 (inflow concentration).
- `c_in::Real`: The initial concentration in the column (t=0).
- `v::Real`: The velocity of the solute.
- `Dl::Real`: The longitudinal dispersion coefficient.
# Returns
    nothing, the results are stored in the `cr` array. 
"""
function constant_injection!(
    cr::Matrix,
    t,
    x,
    c0,
    c_in,
    v,
    Dl,
    )
    @argcheck size(cr, 1) == length(x) && size(cr, 2) == length(t)

    for i in eachindex(t)
        @. cr[:, i] = c_in + (c0 - c_in) / 2 * erfc((x - v * t[i])
         / (2 * sqrt.(Dl * t[i])))
    end
    return nothing
end

"""
pulse_injection(cr, x, t, c0, c_in, v, Dl, t_pulse)

Calculates the concentration profile of a solute undergoing
advection-dispersion transport in a porous 1D domain with pulse injection
(starts at t=0 and ends at t=t_pulse)
Original reference: (Ogata & Banks, 1961):
Appelo, C.A.J.; Postma, Dieke. Geochemistry, Groundwater and Pollution (p. 104).

# Arguments
- `cr::Matrix`: A 2D array to store the concentration profile of a solute. 
    first dimension is time and second dimension is space.
- `x::Real`: location. (Where to sample the concentration)
- `t::Vector`: A 1D array of time locations. (When to sample the concentration)
- `c0::Real`: The concentratio at x=0 (inflow concentration).
- `c_in::Real`: The initial concentration in the column (t=0).
- `v::Real`: The velocity of the solute.
- `Dl::Real`: The longitudinal dispersion coefficient.
- `t_pulse::Real`: The time at which the pulse injection ends (at x=0).
"""
function pulse_injection!(
    cr::Matrix,
    t,
    x,
    c0,
    c_in,
    v,
    Dl,
    t_pulse
    )
    @argcheck size(cr, 1) == length(x) && size(cr, 2) == length(t)
    ratio = (c0 - c_in) / 2
    for j in eachindex(x), i in eachindex(t)
        cr[j, i] = c_in .+ ratio .* (erfc.((x[j] .- v .* t[i])
        ./ (2 .* sqrt.(Dl .* t[i]))))
        if t[i] > t_pulse
            cr[j, i] -= ratio .* (erfc.((x[j] .- v .* (t[i] .- t_pulse))
            ./ (2 .* sqrt.(Dl .* (t[i] .- t_pulse)))))
        end
    end
    return nothing
end

