from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution, minimize
import matplotlib.pyplot as plt


@dataclass
class LocalState:
    """Single-sublattice Gutzwiller state and observables."""

    ket: np.ndarray
    probabilities: np.ndarray
    psi: float
    rho: float
    n2_minus_n: float
    variance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "ket": self.ket.copy(),
            "probabilities": self.probabilities.copy(),
            "psi": self.psi,
            "rho": self.rho,
            "n2_minus_n": self.n2_minus_n,
            "variance": self.variance,
        }


@dataclass
class BranchResult:
    branch: str
    odd: LocalState
    even: LocalState
    sigma_psi: float
    delta_rho: float
    energy_per_site: float
    onsite_term: float
    kinetic_term: float
    fluctuation_term: float
    imbalance_term: float
    odd_energy: float
    even_energy: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "odd": self.odd.as_dict(),
            "even": self.even.as_dict(),
            "psi_O": self.odd.psi,
            "psi_E": self.even.psi,
            "rho_O": self.odd.rho,
            "rho_E": self.even.rho,
            "psi_O_state": self.odd.ket.copy(),
            "psi_E_state": self.even.ket.copy(),
            "Sigma_psi": self.sigma_psi,
            "Delta_rho": self.delta_rho,
            "energy_per_site": self.energy_per_site,
            "onsite_term": self.onsite_term,
            "kinetic_term": self.kinetic_term,
            "fluctuation_term": self.fluctuation_term,
            "imbalance_term": self.imbalance_term,
            "E_O": self.odd_energy,
            "E_E": self.even_energy,
        }


@dataclass
class MeanFieldResult:
    rho_target: float
    zt0_over_u: float
    t0_over_u: float
    geff_ns_over_u: float
    selected_branch: str
    solution: BranchResult
    balanced: BranchResult
    unbalanced: BranchResult

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["solution"] = self.solution.as_dict()
        out["balanced"] = self.balanced.as_dict()
        out["unbalanced"] = self.unbalanced.as_dict()
        return out



def _occupation_arrays(n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = np.arange(n_max + 1, dtype=float)
    n2_minus_n = n * (n - 1.0)
    n2 = n * n
    hop = np.sqrt(np.arange(1, n_max + 1, dtype=float))
    return n, n2_minus_n, n2, hop



def _normalize_probabilities(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    norm = p.sum()
    if norm <= 0.0:
        raise ValueError("Probability vector has zero norm.")
    return p / norm



def local_state_from_probabilities(p: np.ndarray) -> LocalState:
    p = _normalize_probabilities(p)
    n_max = len(p) - 1
    n, n2_minus_n, n2, hop = _occupation_arrays(n_max)

    ket = np.sqrt(p)
    rho = float(np.dot(n, p))
    variance = float(np.dot(n2, p) - rho * rho)
    psi = float(np.sum(hop * ket[:-1] * ket[1:]))
    return LocalState(
        ket=ket,
        probabilities=p,
        psi=psi,
        rho=rho,
        n2_minus_n=float(np.dot(n2_minus_n, p)),
        variance=variance,
    )



def energy_density_coupling_branch(
    p_odd: np.ndarray,
    p_even: np.ndarray,
    *,
    t0_over_u: float,
    geff_ns_over_u: float,
    num_sites: int,
    z: int,
    U: float,
    branch_name: str,
) -> BranchResult:
    """
    Energía por sitio para el ansatz de dos subredes con acoplamiento por densidad.

    Además devuelve:
      E_O = <h_O^MF>
      E_E = <h_E^MF>

    definidos de forma consistente con las ecuaciones de estacionariedad, y tales que
      energy_per_site = 0.5 * (E_O + E_E).
    """
    odd = local_state_from_probabilities(p_odd)
    even = local_state_from_probabilities(p_even)

    t0 = t0_over_u * U
    gbar = geff_ns_over_u * U
    g_local = gbar / max(1, num_sites)

    delta_rho = 0.5 * (odd.rho - even.rho)
    sigma_psi = 0.5 * (abs(odd.psi) + abs(even.psi))

    onsite = 0.25 * U * (odd.n2_minus_n + even.n2_minus_n)
    kinetic = -z * t0 * odd.psi * even.psi
    fluctuation = 0.5 * g_local * (odd.variance + even.variance)
    imbalance = gbar * delta_rho**2
    energy_per_site = onsite + kinetic + fluctuation + imbalance

    # Energías locales efectivas por subred, definidas de modo que
    # (E_O + E_E)/2 = energy_per_site.
    onsite_O = 0.5 * U * odd.n2_minus_n
    onsite_E = 0.5 * U * even.n2_minus_n

    kinetic_O = -z * t0 * odd.psi * even.psi
    kinetic_E = -z * t0 * odd.psi * even.psi

    fluct_O = g_local * odd.variance
    fluct_E = g_local * even.variance

    imbalance_O = 2.0 * gbar * delta_rho * odd.rho
    imbalance_E = -2.0 * gbar * delta_rho * even.rho

    odd_energy = onsite_O + kinetic_O + fluct_O + imbalance_O
    even_energy = onsite_E + kinetic_E + fluct_E + imbalance_E

    return BranchResult(
        branch=branch_name,
        odd=odd,
        even=even,
        sigma_psi=float(sigma_psi),
        delta_rho=float(delta_rho),
        energy_per_site=float(energy_per_site),
        onsite_term=float(onsite),
        kinetic_term=float(kinetic),
        fluctuation_term=float(fluctuation),
        imbalance_term=float(imbalance),
        odd_energy=float(odd_energy),
        even_energy=float(even_energy),
    )

def _initial_density_guess(rho: float, n_max: int) -> np.ndarray:
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



def _repair_single_probability_candidate(x: np.ndarray, rho: float, n_values: np.ndarray) -> np.ndarray | None:
    x = _normalize_probabilities(x)
    if abs(np.dot(n_values, x) - rho) > 1e-7:
        return None
    return x



def _repair_double_probability_candidate(x: np.ndarray, rho: float, n_values: np.ndarray) -> np.ndarray | None:
    dim = len(n_values)
    p_odd = _normalize_probabilities(x[:dim])
    p_even = _normalize_probabilities(x[dim:])
    if abs(np.dot(n_values, p_odd) + np.dot(n_values, p_even) - 2.0 * rho) > 1e-7:
        return None
    return np.concatenate([p_odd, p_even])



def _build_unbalanced_seed(rho: float, n_max: int, delta: float) -> tuple[np.ndarray, np.ndarray]:
    rho_odd = np.clip(rho + delta, 0.0, float(n_max))
    rho_even = np.clip(2.0 * rho - rho_odd, 0.0, float(n_max))
    if abs(0.5 * (rho_odd + rho_even) - rho) > 1e-12:
        rho_odd = rho
        rho_even = rho
    return _initial_density_guess(rho_odd, n_max), _initial_density_guess(rho_even, n_max)



def _optimize_balanced_branch(
    rho: float,
    *,
    t0_over_u: float,
    geff_ns_over_u: float,
    n_max: int,
    num_sites: int,
    z: int,
    U: float,
    seed: int,
) -> BranchResult:
    dim = n_max + 1
    n_values = np.arange(dim, dtype=float)
    bounds = [(0.0, 1.0)] * dim

    def objective(p: np.ndarray) -> float:
        return energy_density_coupling_branch(
            p,
            p,
            t0_over_u=t0_over_u,
            geff_ns_over_u=geff_ns_over_u,
            num_sites=num_sites,
            z=z,
            U=U,
            branch_name="balanced",
        ).energy_per_site

    linear_constraint = LinearConstraint(
        np.vstack([np.ones(dim), n_values]),
        lb=np.array([1.0, rho]),
        ub=np.array([1.0, rho]),
    )
    constraints = [
        {"type": "eq", "fun": lambda p: np.sum(p) - 1.0},
        {"type": "eq", "fun": lambda p: np.dot(n_values, p) - rho},
    ]

    rng = np.random.default_rng(seed)
    candidates = [_initial_density_guess(rho, n_max)]
    candidates.extend(rng.dirichlet(np.ones(dim), size=8))

    best_x = None
    best_energy = np.inf

    try:
        de = differential_evolution(
            objective,
            bounds=bounds,
            constraints=(linear_constraint,),
            strategy="best1bin",
            maxiter=80,
            popsize=10,
            tol=1e-7,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            seed=seed,
            updating="deferred",
            workers=1,
        )
        if de.success:
            x = _repair_single_probability_candidate(de.x, rho, n_values)
            if x is not None:
                best_x = x
                best_energy = objective(x)
                candidates.insert(0, x)
    except Exception:
        pass

    for x0 in candidates:
        try:
            res = minimize(
                objective,
                x0=np.asarray(x0, dtype=float),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-7, "disp": False},
            )
        except Exception:
            continue
        if not res.success:
            continue
        x = _repair_single_probability_candidate(res.x, rho, n_values)
        if x is None:
            continue
        energy = objective(x)
        if energy < best_energy:
            best_x = x
            best_energy = energy

    if best_x is None:
        raise RuntimeError("Balanced branch optimization failed.")

    return energy_density_coupling_branch(
        best_x,
        best_x,
        t0_over_u=t0_over_u,
        geff_ns_over_u=geff_ns_over_u,
        num_sites=num_sites,
        z=z,
        U=U,
        branch_name="balanced",
    )



def _optimize_unbalanced_branch(
    rho: float,
    *,
    t0_over_u: float,
    geff_ns_over_u: float,
    n_max: int,
    num_sites: int,
    z: int,
    U: float,
    seed: int,
) -> BranchResult:
    dim = n_max + 1
    total_dim = 2 * dim
    n_values = np.arange(dim, dtype=float)
    bounds = [(0.0, 1.0)] * total_dim

    def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(x[:dim], dtype=float), np.asarray(x[dim:], dtype=float)

    def objective(x: np.ndarray) -> float:
        p_odd, p_even = split(x)
        return energy_density_coupling_branch(
            p_odd,
            p_even,
            t0_over_u=t0_over_u,
            geff_ns_over_u=geff_ns_over_u,
            num_sites=num_sites,
            z=z,
            U=U,
            branch_name="unbalanced",
        ).energy_per_site

    A = np.zeros((3, total_dim), dtype=float)
    A[0, :dim] = 1.0
    A[1, dim:] = 1.0
    A[2, :dim] = n_values
    A[2, dim:] = n_values
    linear_constraint = LinearConstraint(
        A,
        lb=np.array([1.0, 1.0, 2.0 * rho]),
        ub=np.array([1.0, 1.0, 2.0 * rho]),
    )
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:dim]) - 1.0},
        {"type": "eq", "fun": lambda x: np.sum(x[dim:]) - 1.0},
        {
            "type": "eq",
            "fun": lambda x: np.dot(n_values, x[:dim]) + np.dot(n_values, x[dim:]) - 2.0 * rho,
        },
    ]

    candidates: list[np.ndarray] = []
    base = _initial_density_guess(rho, n_max)
    candidates.append(np.concatenate([base, base]))

    max_delta = min(rho, n_max - rho)
    for frac in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        delta = frac * max_delta
        p_odd, p_even = _build_unbalanced_seed(rho, n_max, delta)
        candidates.append(np.concatenate([p_odd, p_even]))
        candidates.append(np.concatenate([p_even, p_odd]))

    rng = np.random.default_rng(seed)
    for _ in range(10):
        p_odd = rng.dirichlet(np.ones(dim))
        p_even = rng.dirichlet(np.ones(dim))
        if abs(np.dot(n_values, p_odd) + np.dot(n_values, p_even) - 2.0 * rho) > 1e-8:
            q_odd, q_even = _build_unbalanced_seed(rho, n_max, 0.0)
            lam = 0.35
            p_odd = (1.0 - lam) * p_odd + lam * q_odd
            p_even = (1.0 - lam) * p_even + lam * q_even
        candidates.append(np.concatenate([p_odd, p_even]))

    best_x = None
    best_energy = np.inf

    try:
        de = differential_evolution(
            objective,
            bounds=bounds,
            constraints=(linear_constraint,),
            strategy="best1bin",
            maxiter=100,
            popsize=10,
            tol=1e-7,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            seed=seed,
            updating="deferred",
            workers=1,
        )
        if de.success:
            x = _repair_double_probability_candidate(de.x, rho, n_values)
            if x is not None:
                best_x = x
                best_energy = objective(x)
                candidates.insert(0, x)
    except Exception:
        pass

    for x0 in candidates:
        try:
            res = minimize(
                objective,
                x0=np.asarray(x0, dtype=float),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-7, "disp": False},
            )
        except Exception:
            continue
        if not res.success:
            continue
        x = _repair_double_probability_candidate(res.x, rho, n_values)
        if x is None:
            continue
        energy = objective(x)
        if energy < best_energy:
            best_x = x
            best_energy = energy

    if best_x is None:
        raise RuntimeError("Unbalanced branch optimization failed.")

    p_odd, p_even = split(best_x)
    return energy_density_coupling_branch(
        p_odd,
        p_even,
        t0_over_u=t0_over_u,
        geff_ns_over_u=geff_ns_over_u,
        num_sites=num_sites,
        z=z,
        U=U,
        branch_name="unbalanced",
    )


def solve_mean_field(
    rho: float,
    zt0_over_u: float,
    *,
    geff_ns_over_u: float = -0.5,
    n_max: int = 6,
    num_sites: int = 100,
    z: int = 4,
    U: float = 1.0,
    seed: int = 1234,
) -> dict[str, Any]:

    t0_over_u = zt0_over_u / z

    balanced = _optimize_balanced_branch(
        rho,
        t0_over_u=t0_over_u,
        geff_ns_over_u=geff_ns_over_u,
        n_max=n_max,
        num_sites=num_sites,
        z=z,
        U=U,
        seed=seed,
    )
    unbalanced = _optimize_unbalanced_branch(
        rho,
        t0_over_u=t0_over_u,
        geff_ns_over_u=geff_ns_over_u,
        n_max=n_max,
        num_sites=num_sites,
        z=z,
        U=U,
        seed=seed + 17,
    )

    solution = unbalanced if unbalanced.energy_per_site < balanced.energy_per_site - 1e-7 else balanced
    result = MeanFieldResult(
        rho_target=float(rho),
        zt0_over_u=float(zt0_over_u),
        t0_over_u=float(t0_over_u),
        geff_ns_over_u=float(geff_ns_over_u),
        selected_branch=solution.branch,
        solution=solution,
        balanced=balanced,
        unbalanced=unbalanced,
    )
    return result.as_dict()

def plot_phase_diagram(zt_vals, rho_vals):
    Delta_rho = np.zeros((len(rho_vals), len(zt_vals)))
    Sigma_psi = np.zeros((len(rho_vals), len(zt_vals)))

    # Loop
    for i, rho in enumerate(rho_vals):
        for j, zt in enumerate(zt_vals):
            print(i,j)
            res = solve_mean_field(
                rho=rho,
                zt0_over_u=zt,
                geff_ns_over_u=-0.5,
                n_max=6,
                num_sites=100,
                z=4,
                U=1.0,
                seed=1234,
            )
            Delta_rho[i, j] = abs(res["solution"]["Delta_rho"])
            Sigma_psi[i, j] = res["solution"]["Sigma_psi"]

    # Plot Delta_rho
    plt.figure()
    plt.imshow(Delta_rho, origin="lower", aspect="auto",
            extent=[zt_vals.min(), zt_vals.max(), rho_vals.min(), rho_vals.max()])
    plt.xlabel("zt/U")
    plt.ylabel("rho")
    plt.title("Delta rho")
    plt.colorbar()

    # Plot Sigma_psi
    plt.figure()
    plt.imshow(Sigma_psi, origin="lower", aspect="auto",
            extent=[zt_vals.min(), zt_vals.max(), rho_vals.min(), rho_vals.max()])
    plt.xlabel("zt/U")
    plt.ylabel("rho")
    plt.title("Sigma psi")
    plt.colorbar()

    plt.show()


if __name__ == "__main__":

    zt_vals = np.linspace(0.0, 0.3, 61)
    rho_vals = np.linspace(0.0, 3.0, 61)
    # plot_phase_diagram(zt_vals, rho_vals)
    
    rho_example = 0.5
    zt0 = 0.10
    t0=zt0/4
    ns = 100
    geff = -0.5
    result = solve_mean_field(
        rho=rho_example,
        zt0_over_u=zt0,
        geff_ns_over_u=geff,
        n_max=3,
        num_sites=ns,
        z=4,
        U=1.0,
        seed=1234,
    )
    print(result["solution"]["psi_O_state"])
    print(result["solution"]["psi_E_state"])
    print(result["solution"]["E_O"])
    print(result["solution"]["E_E"])

