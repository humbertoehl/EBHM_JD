import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def occupation_arrays(n_max):
    n = np.arange(n_max + 1, dtype=float)
    n2_minus_n = n * (n - 1.0)
    n2 = n * n
    hop = np.sqrt(np.arange(1, n_max + 1, dtype=float))
    return n, n2_minus_n, n2, hop


def normalize_probabilities(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    norm = p.sum()
    if norm <= 0.0:
        raise ValueError("El vector de probabilidades tiene norma cero.")
    return p / norm


def initial_density_guess(rho, n_max):
    guess = np.zeros(n_max + 1, dtype=float)
    n0 = max(0, min(int(math.floor(rho)), n_max))
    n1 = min(n0 + 1, n_max)
    frac = rho - n0
    if n0 == n1:
        guess[n0] = 1.0
    else:
        guess[n0] = 1.0 - frac
        guess[n1] = frac
    return guess


def local_observables(probabilities):
    p = normalize_probabilities(probabilities)
    n_max = len(p) - 1
    n, n2_minus_n, n2, hop = occupation_arrays(n_max)

    ket = np.sqrt(p)
    rho = float(np.dot(n, p))
    variance = float(np.dot(n2, p) - rho * rho)
    psi = float(np.sum(hop * ket[:-1] * ket[1:]))
    onsite_density = float(np.dot(n2_minus_n, p))

    return ket, p, psi, rho, onsite_density, variance


def evaluate_two_sublattice_energy(
    p_odd,
    p_even,
    t0_over_u,
    geff_ns_over_u,
    num_sites,
    z,
    U,
):
    ket_odd, p_odd, psi_odd, rho_odd, onsite_odd, var_odd = local_observables(p_odd)
    ket_even, p_even, psi_even, rho_even, onsite_even, var_even = local_observables(p_even)

    t0 = t0_over_u * U
    gbar = geff_ns_over_u * U
    g_local = gbar / max(1, num_sites)

    delta_rho = 0.5 * (rho_odd - rho_even)
    sigma_psi = 0.5 * (abs(psi_odd) + abs(psi_even))

    onsite = 0.25 * U * (onsite_odd + onsite_even)
    kinetic = -z * t0 * psi_odd * psi_even
    fluctuation = 0.5 * g_local * (var_odd + var_even)
    imbalance = gbar * delta_rho**2
    energy_per_site = onsite + kinetic + fluctuation + imbalance

    # Energías locales efectivas por subred
    e_odd = (
        0.5 * U * onsite_odd
        - z * t0 * psi_odd * psi_even
        + g_local * var_odd
        + 2.0 * gbar * delta_rho * rho_odd
    )
    e_even = (
        0.5 * U * onsite_even
        - z * t0 * psi_odd * psi_even
        + g_local * var_even
        - 2.0 * gbar * delta_rho * rho_even
    )

    return (
        energy_per_site,
        ket_even,
        ket_odd,
        psi_even,
        psi_odd,
        rho_even,
        rho_odd,
        e_even,
        e_odd,
        sigma_psi,
        abs(delta_rho),
        p_even,
        p_odd,
    )


def optimize_balanced_branch(rho, t0_over_u, geff_ns_over_u, n_max, num_sites, z, U, seed):
    dim = n_max + 1
    n_values = np.arange(dim, dtype=float)
    bounds = [(0.0, 1.0)] * dim

    def objective(p):
        return evaluate_two_sublattice_energy(
            p,
            p,
            t0_over_u=t0_over_u,
            geff_ns_over_u=geff_ns_over_u,
            num_sites=num_sites,
            z=z,
            U=U,
        )[0]

    constraints = [
        {"type": "eq", "fun": lambda p: np.sum(p) - 1.0},
        {"type": "eq", "fun": lambda p: np.dot(n_values, p) - rho},
    ]

    rng = np.random.default_rng(seed)
    seeds = [initial_density_guess(rho, n_max)]
    seeds.extend(rng.dirichlet(np.ones(dim), size=10))

    best_energy = np.inf
    best_p = None

    for x0 in seeds:
        result = minimize(
            objective,
            x0=np.asarray(x0, dtype=float),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-8, "disp": False},
        )
        if not result.success:
            continue

        p = normalize_probabilities(result.x)
        if abs(np.dot(n_values, p) - rho) > 1e-6:
            continue

        energy = objective(p)
        if energy < best_energy:
            best_energy = energy
            best_p = p

    if best_p is None:
        raise RuntimeError("Falló la optimización de la rama balanceada.")

    return evaluate_two_sublattice_energy(
        best_p,
        best_p,
        t0_over_u=t0_over_u,
        geff_ns_over_u=geff_ns_over_u,
        num_sites=num_sites,
        z=z,
        U=U,
    )


def optimize_unbalanced_branch(rho, t0_over_u, geff_ns_over_u, n_max, num_sites, z, U, seed):
    dim = n_max + 1
    total_dim = 2 * dim
    n_values = np.arange(dim, dtype=float)
    bounds = [(0.0, 1.0)] * total_dim

    def split(x):
        return np.asarray(x[:dim], dtype=float), np.asarray(x[dim:], dtype=float)

    def objective(x):
        p_odd, p_even = split(x)
        return evaluate_two_sublattice_energy(
            p_odd,
            p_even,
            t0_over_u=t0_over_u,
            geff_ns_over_u=geff_ns_over_u,
            num_sites=num_sites,
            z=z,
            U=U,
        )[0]

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:dim]) - 1.0},
        {"type": "eq", "fun": lambda x: np.sum(x[dim:]) - 1.0},
        {
            "type": "eq",
            "fun": lambda x: np.dot(n_values, x[:dim]) + np.dot(n_values, x[dim:]) - 2.0 * rho,
        },
    ]

    rng = np.random.default_rng(seed)
    seeds = []

    base = initial_density_guess(rho, n_max)
    seeds.append(np.concatenate([base, base]))

    max_delta = min(rho, n_max - rho)
    for frac in (0.15, 0.30, 0.50, 0.75, 1.0):
        delta = frac * max_delta
        rho_odd = np.clip(rho + delta, 0.0, float(n_max))
        rho_even = np.clip(2.0 * rho - rho_odd, 0.0, float(n_max))
        p_odd = initial_density_guess(rho_odd, n_max)
        p_even = initial_density_guess(rho_even, n_max)
        seeds.append(np.concatenate([p_odd, p_even]))
        seeds.append(np.concatenate([p_even, p_odd]))

    for _ in range(10):
        p_odd = rng.dirichlet(np.ones(dim))
        p_even = rng.dirichlet(np.ones(dim))
        seeds.append(np.concatenate([p_odd, p_even]))

    best_energy = np.inf
    best_x = None

    for x0 in seeds:
        result = minimize(
            objective,
            x0=np.asarray(x0, dtype=float),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1200, "ftol": 1e-8, "disp": False},
        )
        if not result.success:
            continue

        x = np.asarray(result.x, dtype=float)
        p_odd, p_even = split(x)
        p_odd = normalize_probabilities(p_odd)
        p_even = normalize_probabilities(p_even)

        avg_rho = 0.5 * (np.dot(n_values, p_odd) + np.dot(n_values, p_even))
        if abs(avg_rho - rho) > 1e-6:
            continue

        energy = objective(np.concatenate([p_odd, p_even]))
        if energy < best_energy:
            best_energy = energy
            best_x = np.concatenate([p_odd, p_even])

    if best_x is None:
        raise RuntimeError("Falló la optimización de la rama desbalanceada.")

    p_odd, p_even = split(best_x)
    return evaluate_two_sublattice_energy(
        p_odd,
        p_even,
        t0_over_u=t0_over_u,
        geff_ns_over_u=geff_ns_over_u,
        num_sites=num_sites,
        z=z,
        U=U,
    )


def solve_mean_field(
    rho,
    zt0_over_u,
    geff_ns_over_u=-0.5,
    n_max=6,
    num_sites=100,
    z=4,
    U=1.0,
    seed=1234,
):
    """
    Resuelve balanceado y desbalanceado, y devuelve la solución de menor energía.

    Salida (tupla):
    (estado_par, estado_impar, psi_par, psi_impar, rho_par, rho_impar,
     Energy_par, Energy_impar, SigmaPsi, abs(DeltaRho), rama_seleccionada)
    """
    t0_over_u = zt0_over_u / z

    balanced = optimize_balanced_branch(rho, t0_over_u, geff_ns_over_u, n_max, num_sites, z, U, seed)
    unbalanced = optimize_unbalanced_branch(
        rho,
        t0_over_u,
        geff_ns_over_u,
        n_max,
        num_sites,
        z,
        U,
        seed + 31,
    )

    # índice 0 = energía por sitio total
    selected = unbalanced if unbalanced[0] < balanced[0] - 1e-8 else balanced
    selected_branch = "unbalanced" if selected is unbalanced else "balanced"

    return (
        selected[1],  # estado_par (ket even)
        selected[2],  # estado_impar (ket odd)
        selected[3],  # psi_par
        selected[4],  # psi_impar
        selected[5],  # rho_par
        selected[6],  # rho_impar
        selected[7],  # Energy_par
        selected[8],  # Energy_impar
        selected[9],  # SigmaPsi
        selected[10],  # abs(DeltaRho)
        selected_branch,
    )


def plot_phase_diagrams(
    zt_values,
    rho_values,
    geff_ns_over_u=-0.5,
    n_max=6,
    num_sites=100,
    z=4,
    U=1.0,
    seed=1234,
):
    sigma_map = np.zeros((len(rho_values), len(zt_values)), dtype=float)
    delta_map = np.zeros((len(rho_values), len(zt_values)), dtype=float)

    for i, rho in enumerate(rho_values):
        for j, zt in enumerate(zt_values):
            (
                _state_even,
                _state_odd,
                _psi_even,
                _psi_odd,
                _rho_even,
                _rho_odd,
                _e_even,
                _e_odd,
                sigma_psi,
                delta_rho_abs,
                _branch,
            ) = solve_mean_field(
                rho=rho,
                zt0_over_u=zt,
                geff_ns_over_u=geff_ns_over_u,
                n_max=n_max,
                num_sites=num_sites,
                z=z,
                U=U,
                seed=seed,
            )
            sigma_map[i, j] = sigma_psi
            delta_map[i, j] = delta_rho_abs

    plt.figure(figsize=(7, 5))
    plt.imshow(
        sigma_map,
        origin="lower",
        aspect="auto",
        extent=[zt_values.min(), zt_values.max(), rho_values.min(), rho_values.max()],
    )
    plt.xlabel("zt/U")
    plt.ylabel("rho")
    plt.title("Diagrama de fase: SigmaPsi")
    plt.colorbar(label="SigmaPsi")

    plt.figure(figsize=(7, 5))
    plt.imshow(
        delta_map,
        origin="lower",
        aspect="auto",
        extent=[zt_values.min(), zt_values.max(), rho_values.min(), rho_values.max()],
    )
    plt.xlabel("zt/U")
    plt.ylabel("rho")
    plt.title("Diagrama de fase: |DeltaRho|")
    plt.colorbar(label="|DeltaRho|")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    zt_vals = np.linspace(0.0, 0.3, 31)
    rho_vals = np.linspace(0.0, 3.0, 31)

    state_even, state_odd, psi_even, psi_odd, rho_even, rho_odd, e_even, e_odd, sigma_psi, delta_rho, branch = solve_mean_field(
        rho=0.5,
        zt0_over_u=0.10,
        geff_ns_over_u=-0.5,
        n_max=6,
        num_sites=100,
        z=4,
        U=1.0,
        seed=1234,
    )

    print("Rama seleccionada:", branch)
    print("estado_par:", state_even)
    print("estado_impar:", state_odd)
    print("psi_par, psi_impar:", psi_even, psi_odd)
    print("rho_par, rho_impar:", rho_even, rho_odd)
    print("Energy_par, Energy_impar:", e_even, e_odd)
    print("SigmaPsi:", sigma_psi)
    print("abs(DeltaRho):", delta_rho)

    # plot_phase_diagrams(zt_vals, rho_vals)
