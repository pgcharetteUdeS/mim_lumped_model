"""
Optical properties of metallic films for vertical-cavity optoelectronic devices
Rakic, M. et al. J. Appl. Phys. 2000, 87, 1–8.
https://opg.optica.org/ao/abstract.cfm?uri=ao-37-22-5271

"""
__all__ = ["rakic_gold_drude_lorentz_model"]

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import constants as syc


@dataclass
class Oscillator:
    """
    Drude-Lorentz oscillator parameters

    f: strength
    Γ_ev: 1/lifetime (1/eV)
    ω_ev: frequency (eV)
    """

    f: float
    Γ_ev: float
    ω_ev: float


def drude_lorentz(λ) -> tuple[float, float, float, float]:
    """
    Lorentz-Drude model of the dielectric function of gold

    Args:
        λ (float):wavelength (m)

    Returns: n, k, ωp, τ

    """

    # Convert input wavelength (m) to energy (eV)
    ω_ev = wavalength_in_meters_to_energy_in_ev(λ)

    # Fundamental oscillator
    ωp_ev: float = 9.03  # eV
    f0: float = 0.760
    Γ0_ev: float = 0.053  # eV
    Ωp: float = np.sqrt(f0) * ωp_ev
    ε = 1 - Ωp**2 / (ω_ev * (ω_ev + 1j * Γ0_ev))

    # Higher order oscillators
    oscillators: list = [
        Oscillator(0.024, 0.241, 0.415),
        Oscillator(0.010, 0.345, 0.830),
        Oscillator(0.071, 0.870, 2.969),
        Oscillator(0.601, 2.494, 4.304),
        Oscillator(4.384, 2.214, 13.32),
    ]
    for oscillator in oscillators:
        ε += (
            oscillator.f
            * ωp_ev**2
            / ((oscillator.ω_ev**2 - ω_ev**2) - 1j * ω_ev * oscillator.Γ_ev)
        )

    # Refractive index and extinction coefficient
    n = np.sqrt(ε).real
    k = np.sqrt(ε).imag

    # Plasma frequency and relaxation time
    ωp = 2 * np.pi * energy_in_ev_to_frequency_in_hz(ωp_ev)
    τ = 1 / (energy_in_ev_to_frequency_in_hz(Γ0_ev) * 2 * np.pi)

    return n, k, ωp, τ


def wavalength_in_meters_to_energy_in_ev(λ: float):
    """

    Calculate energy in eV from wavelength in meters

    Args:
        λ (float): wavelength (m)

    Returns: energy in eV

    """

    return (syc.h * syc.c) / (λ * syc.electron_volt)


def energy_in_ev_to_frequency_in_hz(e: float):
    """
    Calculate frequency in Hz from energy in eV

    Args:
        e (float): energy in eV

    Returns:frequency in Hz

    """

    return e * syc.electron_volt / syc.h


def rakic_gold_drude_lorentz_model():
    """
    Drude-Lorentz model of the dielectric function of gold.

    Returns: None

    """

    λ_max: float = 8e-6
    λ_min: float = 3e-6
    n: int = 1000
    λs: np.ndarray = np.linspace(λ_min, λ_max, n)
    n, k, ωp, τ = np.vectorize(drude_lorentz)(λs)

    # plot n,k vs eV
    plt.rc("font", family="Arial", size="14")
    plt.figure()
    plt.plot(λs * 1e6, n, label="n")
    plt.plot(λs * 1e6, k, label="k")
    plt.xlabel("λ (μm)")
    plt.ylabel("n, k")
    plt.ylim(bottom=0)
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0), loc=3, ncol=2, borderaxespad=0)
    plt.grid()
    plt.show()

    # Write data to Excel file
    df1 = pd.DataFrame(
        {
            "A": ["Element symbol", "Plasma frequency (rads/s)", "Relaxation time (s)"],
            "B": ["Au", ωp[0], τ[0]],
        }
    )
    df2 = pd.DataFrame({"wavelength (um)": λs * 1e6, "n": n, "k": k})
    with pd.ExcelWriter("data/Rakic-Au.xlsx") as writer:
        df1.to_excel(writer, sheet_name="properties", index=False, header=False)
        df2.to_excel(writer, sheet_name="n_and_k", index=False)
