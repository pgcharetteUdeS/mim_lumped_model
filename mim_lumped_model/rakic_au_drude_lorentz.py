"""

Dielectric function of gold over a range of wavelengths with the Drude-Lorentz model by:
    Optical properties of metallic films for vertical-cavity optoelectronic devices
    Rakic, et al. J. Appl. Phys. 2000, 87, 1–8.
    https://opg.optica.org/ao/abstract.cfm?uri=ao-37-22-5271

"""
__all__ = ["rakic_au_drude_lorentz"]

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
    Γ: 1/lifetime (1/eV)
    ω: frequency (eV)
    """

    f: float
    Γ: float
    ω: float


def rakic(λ: float) -> tuple[float, float, float, float, float, float]:
    """
    [rakic, 2000] DL model of the dielectric function of gold at wavelength λ
    consisting of a fundamental oscillator and 5 higher order oscillators

    Args:
        λ (float):wavelength (m)

    NB: all frequencies internally in eV

    Returns: n, k, n(fundamental), k(fundamental), ωp, τ

    """

    # Convert input wavelength (m) to energy (eV)
    ω_ev = wavelength_in_meters_to_energy_in_ev(λ)

    # Fundamental oscillator
    ωp: float = 9.03
    f0: float = 0.760
    Γ0: float = 0.053
    Ωp: float = np.sqrt(f0) * ωp
    ε_fundamental: complex = 1 - Ωp**2 / (ω_ev * (ω_ev + 1j * Γ0))
    ε: complex = ε_fundamental

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
            * ωp**2
            / ((oscillator.ω**2 - ω_ev**2) - 1j * ω_ev * oscillator.Γ)
        )

    # Refractive index and extinction coefficients
    n_fundamental = np.sqrt(ε_fundamental).real
    k_fundamental = np.sqrt(ε_fundamental).imag
    n = np.sqrt(ε).real
    k = np.sqrt(ε).imag

    # Plasma frequency and relaxation time
    ωp_rads_per_s = 2 * np.pi * energy_in_ev_to_frequency_in_hz(ωp)
    τ = 1 / (energy_in_ev_to_frequency_in_hz(Γ0) * 2 * np.pi)

    return n, k, n_fundamental, k_fundamental, ωp_rads_per_s, τ


def wavelength_in_meters_to_energy_in_ev(λ: float):
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


def rakic_au_drude_lorentz(λ_min: float, λ_max: float, n: int):
    """

    Drude-Lorentz model of the dielectric function of gold over a range of wavelengths
    using both only the fundamental oscillator and the full model (+5 higher order
    oscillators), write data to Excel files and plot n,k vs wavelength.

    Args:
        λ_min (float): minimum wavelength in the range (m)
        λ_max (float): maximum wavelength in the range (m)
        n (int): number of samples

    Returns: None

    """

    # Calculate n,k over the range of wavelengths
    λs: np.ndarray = np.linspace(λ_min, λ_max, n)
    n, k, n_fundamental, k_fundamental, ωp, τ = np.vectorize(rakic)(λs)

    # Plot n,k vs wavelength
    plt.figure()
    plt.plot(λs * 1e6, n, "g", label="n (full)")
    plt.plot(λs * 1e6, n_fundamental, "g--", label="n (fundamental only)")
    plt.plot(λs * 1e6, k, "b", label="k (full)")
    plt.plot(λs * 1e6, k_fundamental, "b--", label="k (fundamental only)")
    plt.title(
        "[Rakic, 2000] Drude-Lorentz model for Au : "
        "fundamental oscillator with 5 higher order oscillators"
    )
    plt.xlabel("λ (μm)")
    plt.ylabel("n, k")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.show()

    # Write data to Excel files ("Rakic-Au-DL.xlsx" and "Rakic-Au-DL-fundamental.xlsx")
    df1: pd.DataFrame = pd.DataFrame(
        {
            "A": ["Element symbol", "Plasma frequency (rads/s)", "Relaxation time (s)"],
            "B": ["Au", ωp[0], τ[0]],
        }
    )
    df2: pd.DataFrame = pd.DataFrame(
        {"wavelength (um)": λs * 1e6, "n": n_fundamental, "k": k_fundamental}
    )
    with pd.ExcelWriter("data/Rakic-Au-DL-fundamental.xlsx") as writer:
        df1.to_excel(writer, sheet_name="properties", index=False, header=False)
        df2.to_excel(writer, sheet_name="n_and_k", index=False)
    df2 = pd.DataFrame({"wavelength (um)": λs * 1e6, "n": n, "k": k})
    with pd.ExcelWriter("data/Rakic-Au-DL.xlsx") as writer:
        df1.to_excel(writer, sheet_name="properties", index=False, header=False)
        df2.to_excel(writer, sheet_name="n_and_k", index=False)
