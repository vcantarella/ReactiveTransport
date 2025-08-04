include("../src/analytical_functions.jl")

using Test


# benchmark tests
using BenchmarkTools
function test_benchmark_analytical_functions()
    # Define parameters
    x = 0.5  # Spatial locations
    t = range(0.1, stop=72000, length=100)  # Time locations
    c0 = 1.0
    c_in = 0.0
    v = 1e-5
    alpha_l = 1e-2
    Dl = 1e-9 + alpha_l * v
    t_pulse = 36000.0
    cr = zeros(length(x), length(t))
    cr_pulse = zeros(length(x), length(t))

    display(@benchmark constant_injection!($cr, $t, $x, $c0, $c_in, $v, $Dl))
    display(@benchmark pulse_injection!($cr_pulse, $t, $x, $c0, $c_in, $v, $Dl, $t_pulse))
end

test_analytical_functions()

function test_type_stability()
    # Define parameters
    x = 0.5  # Spatial locations
    t = range(0.1, stop=72000, length=100)  # Time locations
    c0 = 1.0
    c_in = 0.0
    v = 1e-5
    alpha_l = 1e-2
    Dl = 1e-9 + alpha_l * v
    t_pulse = 36000.0
    cr = zeros( length(t))
    cr_pulse = zeros(length(t))
    display(@code_warntype constant_injection!(cr, t, x, c0, c_in, v, Dl))
    display(@code_warntype pulse_injection!(cr_pulse, t, x, c0, c_in, v, Dl, t_pulse))
end
test_type_stability()


## Test that constant injection outputs the same result as pulse injection when t < t_pulse
@testset "Analytical Functions" begin
    # Define parameters
    x = 0.5 # Spatial locations
    t =range(0.1, stop=72000, length=100)  # Time locations
    c0 = 1.0
    c_in = 0.0
    v = 1e-5
    alpha_l = 1e-3
    Dl = 1e-9 + alpha_l * v
    t_pulse = 18000
    cr = zeros(length(x), length(t))
    cr_pulse = zeros(length(x), length(t))
    valid_indexes = findall(t .< t_pulse)
    upper_indexes = findall(t .> t_pulse)
    # Calculate concentration profiles
    constant_injection!(cr, t, x, c0, c_in, v, Dl)
    pulse_injection!(cr_pulse, t, x, c0, c_in, v, Dl, t_pulse)
    # Check that the results are the same for t < t_pulse
    @test all(isapprox.(cr[:, valid_indexes], cr_pulse[:, valid_indexes], atol=1e-6))
    # Check that the results are different for t >= t_pulse
    #@test all(cr[:, upper_indexes] .> cr_pulse[:, upper_indexes])
end

# Test differentiability of the analytical functions
using ForwardDiff
@testset "Differentiability" begin
    # Define parameters
    x = 0.5  # Spatial locations
    t = range(0.1, stop=72000, length=100)  # Time locations
    c0 = 1.0
    c_in = 0.0
    v = 1e-5
    alpha_l = 1e-2
    Dl = 1e-9 + alpha_l * v
    t_pulse = 36000.0

    # Test constant injection differentiability
    function const_inj_result(p, t, x, c0, c_in)
        v = p[1]
        Dl = p[2]
        cr = zeros(eltype(p), length(t))
        constant_injection!(cr, t, x, c0, c_in, v, Dl)
        return cr
    end
    jac = ForwardDiff.jacobian(p -> const_inj_result(p, collect(t), x, c0, c_in), [v, Dl])
    @test size(jac) == (length(t), length([v, Dl]))

    # Test pulse injection differentiability
    function pulse_inj_result(p, t, x, c0, c_in, t_pulse)
        v = p[1]
        Dl = p[2]
        cr_pulse = zeros(eltype(p), length(t))
        pulse_injection!(cr_pulse, t, x, c0, c_in, v, Dl, t_pulse)
        return cr_pulse
    end
    jac_pulse = ForwardDiff.jacobian(p -> pulse_inj_result(p, collect(t), x, c0, c_in, t_pulse), [v, Dl])
    @test size(jac_pulse) == (length(t), length([v, Dl]))
end
