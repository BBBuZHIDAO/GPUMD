import math


def cutoff(r: float, rcut: float) -> float:
    x = r / rcut
    return 1.0 - 35.0 * x**4 + 84.0 * x**5 - 70.0 * x**6 + 20.0 * x**7


def test_two_directed_gpumd_terms_reconstruct_matlab_iwp_pair_energy() -> None:
    r = 7.8
    rho_i = 2.1
    rho_j = 1.7
    z0 = 3.1
    alpha = 8.2
    delta = 2.9
    epsilon = 0.018
    c = 0.024
    d = 15.0
    seff = 2.8
    c6 = 18.0
    rcut = 12.0

    exp0 = math.exp(-(alpha / z0) * (r - z0))
    exp_i = math.exp(-(rho_i / delta) ** 2)
    exp_j = math.exp(-(rho_j / delta) ** 2)
    damping = 1.0 + math.exp(-d * (r / seff - 1.0))
    attractive = -c6 / r**6 / damping

    matlab_pair = cutoff(r, rcut) * (
        attractive + exp0 * (epsilon + c * exp_i + c * exp_j)
    )
    directed_i = cutoff(r, rcut) * (
        0.5 * attractive + exp0 * (0.5 * epsilon + c * exp_i)
    )
    directed_j = cutoff(r, rcut) * (
        0.5 * attractive + exp0 * (0.5 * epsilon + c * exp_j)
    )
    assert math.isclose(directed_i + directed_j, matlab_pair, rel_tol=1.0e-14, abs_tol=1.0e-14)
