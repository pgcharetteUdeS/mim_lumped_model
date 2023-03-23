"""mim_lumped_model.py

    Script qui utilise le modèle RLC groupé de l'article "Ultra‐Narrowband Metamaterial
    Absorbers for High Spectral Resolution Infrared Spectroscopy" [Kang, 2019] pour
    modéliser la réponse d'un filtre à base d'un réseau de nano-structures MIM
    en forme de croix.

    Auteur: Paul Charette

    NB:
    1) Pour aller à des longueurs d'onde au delà de 8.25 um, il faudrait utiliser
       l'interpolation en spline cubique (cubic spline) pour les indices
       de l'oxyde car l'interpolation polynomial devient insuffisante.

    Remarques importantes dans la publie:
    1) "When designing an optimized MIM IR absorber with a high spectral selectivity,
        both FWHM and absorption must be considered simultaneously. Since fpeak
        can be independently tuned by b, there exists a set of Λ and a at a given fpeak
        that guarantees a narrow FWHM and a near-unity absorption."

"""
import os
import sys
import time
from collections import namedtuple
from itertools import product
from matplotlib import use as plt_use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as syc
from typing import TypedDict

from rakic import rakic_au_model
from materials_and_geometry import Geometry, Materials


# Script version
__version__: str = "3.1"


# Constants
Constants = namedtuple("Constants", "z0")
constants = Constants(np.sqrt(syc.mu_0 / syc.epsilon_0))


class FilterResponseMetrics(TypedDict):
    """
    Typed dictionary data type for filter_response_metrics() return values
    """

    λ_peak: float
    fwhm: float
    q: float
    λs: np.ndarray
    absorbance: np.ndarray


def c_p(geom: Geometry) -> float:
    """
    Equation 1: MIM array periodic coupling capacitance

    Args:
        geom (Geometry): structure geometry

    Returns: Cp (F)

    """

    return (
        (np.pi * syc.epsilon_0)
        * geom.a
        / np.log(
            2 * (geom.Λ - geom.b) / geom.t_metal
            + np.sqrt((2 * (geom.Λ - geom.b) / geom.t_metal) ** 2 - 1)
        )
    )


def l_m(geom: Geometry) -> float:
    """
    Equation 2: mutual impedance

    Args:
        geom (Geometry): structure geometry

    Returns: Lm (H)

    """

    return (1 / 2) * syc.mu_0 * geom.t_ins * geom.b / geom.a


def c_m(ω: float, mats: Materials, geom: Geometry) -> complex:
    """
    Equation 3: mutual capacitance at frequency ω

    Args:
        ω (float): radial frequency (rads/s)
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: Cm (F)

    """

    return (
        geom.c
        * syc.epsilon_0
        * mats.ε_ins(λ=syc.c * (2 * np.pi / ω))
        * (geom.b / 2)
        * (geom.a / geom.t_ins)
    )


def l_kc(ω: float, mats: Materials, geom: Geometry) -> float:
    """
    Equation 4: metal cross nano-structure kinetic inductance at frequency ω

    Args:
        ω (float): radial frequency (rads/s)
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: Lk,c (H)

    """

    return (geom.c_prime * geom.b / (geom.a * mats.δ(ω=ω))) * (
        1 / (syc.epsilon_0 * mats.ω_p**2)
    )


def r_c(ω: float, mats: Materials, geom: Geometry) -> float:
    """
    Equation 5: metal cross nano-structure resistance at frequency ω

    Args:
        ω (float): radial frequency (rads/s)
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: Rc (ohm)

    """

    return (geom.c_prime * geom.b / (geom.a * mats.δ(ω=ω))) * (1 / mats.σ)


def l_kg(ω: float, mats: Materials) -> float:
    """
    Equation 6: metal ground plane kinetic inductance at frequency ω

    Args:
        ω (float): radial frequency (rads/s)
        mats (Materials): material properties

    Returns: Lk,g (H)

    """

    return (2 / mats.δ(ω=ω)) * (1 / (syc.epsilon_0 * mats.ω_p**2))


def r_g(ω: float, mats: Materials) -> float:
    """
    Equation 7: metal ground plane resistance at frequency ω

    Args:
        ω (float): radial frequency (rads/s)
        mats (Materials): material properties

    Returns: Rg (ohm)

    """

    return (2 / mats.δ(ω=ω)) * (1 / mats.σ)


def z_cross(ω: float, mats: Materials, geom: Geometry) -> complex:
    """
    Equation S9: complex total impedance Zcross at frequency ω

    Args:
        ω (float): radial frequency (rads/s)
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: Zcross (complex)

    """

    # Laplace transform s variable substitution
    s: complex = 1j * ω

    # Equation S10
    z_e: complex = 1 / (s * c_p(geom=geom))

    # Equation S11
    c_m_val: complex = c_m(ω=ω, mats=mats, geom=geom)
    l_m_val: float = l_m(geom=geom)
    l_g: float = l_kg(ω=ω, mats=mats) + l_m_val
    l_c: float = l_kc(ω=ω, mats=mats, geom=geom) + l_m_val
    r_c_val: float = r_c(ω=ω, mats=mats, geom=geom)
    r_g_val: float = r_g(ω=ω, mats=mats)
    z_m: complex = (
        s**3 * c_m_val * l_c * l_g
        + s**2 * c_m_val * (l_c * r_g_val + l_g * r_c_val)
        + s * (c_m_val * r_c_val * r_g_val + 2 * l_c)
        + 2 * r_c_val
    ) / (2 + s**2 * c_m_val * (l_c + l_g) + s * c_m_val * (r_c_val + r_g_val))

    # Equation S9
    return z_e + z_m


def plot_z_cross_spectrum(mats: Materials, geom: Geometry):
    """
    Plot complex impedance components as a function of wavelength

    NB: if Zcross.real is significantly different from Z0 at the zero-crossing
        wavelength of Zcross.imag, the absorbance will not reach unity at fpeak!

    Args:
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: None

    """

    # Zcross spectrum
    z_cross_spectrum: np.ndarray = np.asarray(
        [z_cross(ω=2 * np.pi * (syc.c / λ), mats=mats, geom=geom) for λ in mats.λs]
    )

    # Find min|Zcross| wavelength and max|Zcross|
    z_cross_max: float = np.abs(z_cross_spectrum).max()
    n_min: int = np.abs(z_cross_spectrum).argmin()
    z_cross_min: float = np.abs(z_cross_spectrum)[n_min]
    λ_z_cross_min: float = mats.λs[n_min]

    # Find Zcross.imag zero crossing wavelength and corresponding value of Zcross.real
    n_zc: int = np.absolute(z_cross_spectrum.imag - 0).argmin()
    λ_zero_crossing: float = mats.λs[n_zc]
    z_cross_real_at_λ_zero_crossing: float = z_cross_spectrum[n_zc].real

    # Plot Zcross n & k as a function of wavelength
    fig, [ax1, ax2] = plt.subplots(2)
    ax2r = ax2.twinx()
    fig.suptitle(
        r"Z$_{cross}$(λ)"
        "\n"
        f"a = {geom.a*1e9:.0f} nm, b = {geom.b*1e6:.0f} nm, Λ = {geom.Λ*1e6:.1f} μm\n"
        rf"t$_{{metal}}$ = {geom.t_metal*1e9:.1f} nm, "
        rf"t$_{{ins}}$ = {geom.t_ins * 1e9:.1f} nm",
    )
    ax1.plot(mats.λs * 1e6, np.abs(z_cross_spectrum))
    ax1.plot(
        [λ_z_cross_min * 1e6, λ_z_cross_min * 1e6],
        [z_cross_min, z_cross_max],
        "r--",
    )
    ax1.annotate(
        rf"min(|Z$_{{cross}}$|) = {z_cross_min:.1f} $\Omega$ "
        f"at λ = {λ_z_cross_min*1e6:.2f} μm",
        xy=(λ_z_cross_min * 1e6, z_cross_min),
        xytext=(λ_z_cross_min * 1e6 + 0.25, z_cross_max / 2),
        arrowprops={"arrowstyle": "->", "color": "black"},
    )
    ax1.set(ylabel=r"|Z$_{cross}$| ($\Omega$)")
    ax1.grid()
    ax2.plot(mats.λs * 1e6, z_cross_spectrum.real, "b")
    ax2r.plot(mats.λs * 1e6, z_cross_spectrum.imag, "r")
    ax2r.plot(
        [λ_zero_crossing * 1e6, λ_zero_crossing * 1e6],
        [z_cross_spectrum.imag.min(), z_cross_spectrum.imag.max()],
        "r--",
    )
    ax2.annotate(
        rf"Z$_{{cross}}$.real = {z_cross_real_at_λ_zero_crossing:.0f} $\Omega$ at the"
        rf" zero crossing of Z$_{{cross}}$.imag (λ = {λ_zero_crossing*1e6:.2f} μm)"
        "\n"
        rf"NB: if Z$_{{cross}}$.real $\neq$ Z$_0$ ({constants.z0:.0f} $\Omega$), "
        "absorbance at f$_{{peak}}$ will not reach unity",
        xy=(λ_zero_crossing * 1e6, z_cross_spectrum.real[n_zc]),
        xytext=(λ_zero_crossing * 1e6 + 0.25, z_cross_max / 5),
        arrowprops={"arrowstyle": "->", "color": "black"},
    )
    ax2.set(xlabel="Wavelength (μm)", ylabel=r"Z$_{cross}$.real ($\Omega$)")
    ax2r.set_ylabel(r"Z$_{cross}$.imag ($\Omega$)", color="r")
    ax2r.tick_params(axis="y", labelcolor="r")
    ax2.grid()

    return None


def absorbance(λ: float, mats: Materials, geom: Geometry) -> float:
    """
    Absorbance at wavelength λ

    Args:
        λ (float): wavelength (m)
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: Absorbance (normalized)

    """

    ω: float = 2 * np.pi * (syc.c / λ)
    z_cross_val: complex = z_cross(ω=ω, mats=mats, geom=geom)
    reflectance: float = (
        np.abs((z_cross_val - constants.z0) / (z_cross_val + constants.z0)) ** 2
    )

    return 1 - reflectance


def plot_absorbance_spectra(
    mats: Materials, geom: Geometry, geometries: np.ndarray, title: str, filename: str
) -> list:
    """
    Combined absorbance spectrum plots for MIMs of different geometries

    Args:
        mats (Materials): material properties
        geom (Geometry): reference structure geometry
        geometries (np.ndarray): MIMs geometries as array of [a, b, Λ] value triplets
        title (str): plot title
        filename (str): filename for saving the plot

    Returns: list of dictionaries of absorbers

    """

    # Convert geometry "array of arrays" to "list of dictionaries", for code clarity
    absorbers: list = [dict(zip(["a", "b", "Λ"], g)) for g in geometries]

    # Loop to plot the absorbance for the different geometries
    fig, ax = plt.subplots()
    for absorber in absorbers:
        response_metrics: FilterResponseMetrics = filter_response_metrics(
            mats=mats,
            geom=geom,
            a=absorber["a"],
            b=absorber["b"],
            Λ=absorber["Λ"],
        )
        absorber["metrics"] = response_metrics
        ax.plot(
            mats.λs * 1e6,
            response_metrics["absorbance"],
            label=rf"λ$_{{peak}}$={response_metrics['λ_peak']*1e6:.2f} μm, "
            f"FWHM={response_metrics['fwhm']*1e9:.0f} nm, "
            f"Q={response_metrics['q']:.1e}\n"
            f"a={absorber['a']*1e9:.0f} nm, "
            f"b={absorber['b']*1e6:.1f} μm, "
            f"Λ={absorber['Λ']*1e6:.1f} μm",
        )
    ax.set(
        title=f"{title}\n"
        f"{mats.metal_name} ({mats.metal_datafile}) : "
        rf"ω$_p$ = 2$\pi$ {mats.ω_p/(2*np.pi):.2e} Hz, τ = {mats.τ:.2e} s, "
        rf"t$_{{metal}}$ = {geom.t_metal*1e9:.1f} nm, "
        f"c = {geom.c: .2f}\n"
        f"{mats.insulator_name} ({mats.insulator_datafile}) : "
        rf"t$_{{ins}}$ = {geom.t_ins * 1e9:.1f} nm",
        xlabel="Wavelength (μm)",
        ylabel="Absorbance",
        ylim=([0, 1]),
    )
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig(f"output/{filename}")

    return absorbers


def filter_response_metrics(
    mats: Materials,
    geom: Geometry,
    a: float,
    b: float,
    Λ: float,
) -> FilterResponseMetrics:
    """
    Filter response metrics λpeak, FWHM, Q, and absorbance spectrum as a function
    of structure geometry parameters a, b, and Λ

    Args:
        mats (Materials): material properties
        geom (Geometry): reference structure geometry
        a (float): cross arm width (m)
        b (float): cross arm length (m)
        Λ (float): cross pattern period (m)

    Returns: λ_peak (m), FWHM (m), Q, absorbance spectrum

    """

    # Load geometry parameters into the reference geometry object
    geom.a = a
    geom.b = b
    geom.Λ = Λ

    # Absorbance as a function of wavelength (absorbance spectrum)
    absorbance_spectrum: np.ndarray = np.asarray(
        [absorbance(λ=λ, mats=mats, geom=geom) for λ in mats.λs]
    )

    # Find absorbance peak wavelength
    λ_peak_index = int(absorbance_spectrum.argmax())
    λ_peak: float = mats.λs[λ_peak_index]

    # Determine FWHM numerically from the absorbance spectrum
    i_left: int = (
        np.absolute(absorbance_spectrum[:λ_peak_index] - 0.5).argmin()
        if len(absorbance_spectrum[:λ_peak_index]) > 0
        else 0
    )
    i_right: int = (
        np.absolute(absorbance_spectrum[λ_peak_index:] - 0.5).argmin()
        if len(absorbance_spectrum[λ_peak_index:]) > 0
        else 0
    )
    fwhm: float = mats.λs[λ_peak_index + i_right] - mats.λs[i_left]

    # Q
    q: float = λ_peak / fwhm

    # Return filter metrics and absorbance spectrum
    return {
        "λ_peak": λ_peak,
        "fwhm": fwhm,
        "q": q,
        "absorbance": absorbance_spectrum,
        "λs": mats.λs,
    }


def figure_2d(mats: Materials, geom: Geometry) -> list:
    """
    Plot Figure 2d from the paper

    Args:
        mats (Materials): material properties
        geom (Geometry): reference structure geometry

    Returns: list of dictionaries of absorbers

    """

    # MIM geometries: triplets of a(m), b (m), Λ(m)
    geometries: np.ndarray = np.asarray(
        [
            [150, 1.5, 3.6],
            [200, 1.7, 3.8],
            [300, 1.9, 4.0],
            [350, 2.1, 4.2],
        ]
    ) * [1e-9, 1e-6, 1e-6]
    absorbers: list = plot_absorbance_spectra(
        mats=mats,
        geom=geom,
        geometries=geometries,
        title="Figure 2d : Absorbance (λ)",
        filename="figure_2d.png",
    )

    return absorbers


def figure_3a(mats: Materials, geom: Geometry) -> list:
    """
    Plot Figure 3a from the paper

    Args:
        mats (Materials): material properties
        geom (Geometry): reference structure geometry

    Returns: None

    """

    # MIM geometries: triplets of a(m), b (m), Λ(m)
    geometries: np.ndarray = np.asarray(
        [
            [150, 1.3, 3.0],
            [150, 1.4, 3.2],
            [200, 1.5, 3.4],
            [200, 1.6, 3.6],
            [200, 1.7, 3.6],
            [200, 1.8, 3.8],
            [200, 1.9, 4.0],
            [300, 2.0, 4.0],
            [300, 2.1, 4.0],
            [300, 2.2, 4.0],
            [300, 2.3, 4.0],
            [300, 2.4, 4.0],
        ]
    ) * [1e-9, 1e-6, 1e-6]
    absorbers: list = plot_absorbance_spectra(
        mats=mats,
        geom=geom,
        geometries=geometries,
        title="Figure 3a : Absorbance(λ) for the 12 MIM IR absorbers",
        filename="figure_3a.png",
    )

    return absorbers


def figure_3b(mats: Materials, geom: Geometry):
    """
    Plot Figure 3b from the paper (λpeak as a function of b), as well as
    a 2D map of FWHM(Λ, a)

    Args:
        mats (Materials): material properties
        geom (Geometry): reference structure geometry

    Returns: None

    """

    # Arrays of cross widths (a), lengths (b) and periods (Λ) for the MIM structures
    n: int = 10
    a_array: np.ndarray = np.linspace(150, 350, n) * 1e-9
    b_array: np.ndarray = np.linspace(1.5, 2.4, n) * 1e-6
    Λ_array: np.ndarray = np.linspace(3.6, 4.2, n) * 1e-6

    # Figure 3b (λpeak as a function of b)
    a_fixed: float = 200e-9
    Λ_fixed: float = 4e-6
    λ_peak_array = np.asarray(
        [
            filter_response_metrics(
                mats=mats,
                geom=geom,
                a=a_fixed,
                b=b,
                Λ=Λ_fixed,
            )["λ_peak"]
            for b in b_array
        ]
    )
    fig, ax = plt.subplots()
    ax.plot(b_array * 1e6, λ_peak_array * 1e6)
    ax.set(
        title=rf"Figure 3b : λ$_{{peak}}$ (b) @ "
        f"a = {a_fixed*1e9:.0f} nm, Λ = {Λ_fixed*1e6:.1f} μm",
        xlabel="b (μm)",
        ylabel="λ$_{peak}$ (μm)",
        ylim=(3, 8),
    )
    ax.grid()

    # 2D map of FWHM(Λ, a)
    b_fixed: float = 1.8e-6
    filter_metrics: FilterResponseMetrics = filter_response_metrics(
        mats=mats,
        geom=geom,
        a=a_fixed,
        b=b_fixed,
        Λ=Λ_fixed,
    )
    λ_peak_fixed = filter_metrics["λ_peak"]
    fwhm_array = np.asarray(
        [
            filter_response_metrics(
                mats=mats,
                geom=geom,
                a=a,
                b=b_fixed,
                Λ=Λ,
            )["fwhm"]
            for Λ, a in product(Λ_array, a_array)
        ]
    ).reshape((len(Λ_array), len(a_array)))
    fig, ax = plt.subplots()
    im = ax.imshow(
        np.flipud(fwhm_array) * 1e9,
        aspect="auto",
        interpolation="bilinear",
        extent=[
            a_array[0] * 1e9,
            a_array[-1] * 1e9,
            Λ_array[0] * 1e6,
            Λ_array[-1] * 1e6,
        ],
    )
    ax.set(
        title=f"FWHM(Λ, a) @ b = {b_fixed*1e6:.1f} μm"
        rf" (λ$_{{peak}}$ = {λ_peak_fixed*1e6:.2f} μm)",
        xlabel="a (nm)",
        ylabel="Λ (μm)",
    )
    fig.colorbar(im, label="FWHM (nm)")

    return None


def save_results_to_file(
    filename: str, absorbers: list, mats: Materials, geom: Geometry
):
    """
    Save geometry, materials and results from a modeling run to an Excel file

    Args:
        filename (str): output filename
        absorbers (list): list of absorbers
        mats (Materials): material properties
        geom (Geometry): reference structure geometry

    Returns: None

    """

    # Check if the Excel output file is already open
    try:
        with open(f"output/{filename}", "a"):
            pass
    except IOError:
        raise IOError(
            f"Could not open 'output/{filename}', close it " "if it's already open!"
        ) from None

    # Write geometry, materials and results to Excel file
    with pd.ExcelWriter(f"output/{filename}") as writer:
        # Write geometry
        df: pd.DataFrame = pd.DataFrame(
            {
                "A": [
                    "a (m)",
                    "b (m)",
                    "c",
                    "Λ (m)",
                    "t_metal (m)",
                    "t_ins (m)",
                ],
                "B": [geom.a, geom.b, geom.c, geom.Λ, geom.t_metal, geom.t_ins],
            }
        )
        df.to_excel(writer, sheet_name="Geometry", index=False, header=False)

        # Write materials
        df = pd.DataFrame(
            {
                "A": [
                    "insulator_datafile",
                    "insulator_εr_r_model_order",
                    "insulator_εr_i_model_order",
                    "metal_datafile",
                    "metal_n_model_order",
                    "metal_κ_model_order",
                    "absorbance_spectrum_sample_count",
                ],
                "B": [
                    mats.insulator_datafile,
                    mats.insulator_εr_r_model_order,
                    mats.insulator_εr_i_model_order,
                    mats.metal_datafile,
                    mats.metal_n_model_order,
                    mats.metal_κ_model_order,
                    mats.absorbance_spectrum_sample_count,
                ],
            }
        )
        df.to_excel(writer, sheet_name="Materials", index=False, header=False)

        # Loop to write results for each absorber
        for i, absorber in enumerate(absorbers):
            df = pd.DataFrame(
                {
                    "A": [
                        "λ_peak (m)",
                        "fwhm (m)",
                        "Q",
                    ],
                    "B": [
                        absorber["metrics"]["λ_peak"],
                        absorber["metrics"]["fwhm"],
                        absorber["metrics"]["q"],
                    ],
                }
            )
            df.to_excel(
                writer, sheet_name=f"osc {i} - properties", index=False, header=False
            )
            df = pd.DataFrame(
                {
                    "wavelength (m)": absorber["metrics"]["λs"],
                    "absorbance": absorber["metrics"]["absorbance"],
                }
            )
            df.to_excel(writer, sheet_name=f"osc {i} - absorbance", index=False)


def main():
    """
    Main calling function

    Returns: None

    """

    # Show Python interpreter version, script name & version on console
    python_version = (
        f"{str(sys.version_info.major)}"
        f".{str(sys.version_info.minor)}"
        f".{str(sys.version_info.micro)}"
    )
    print(
        f"{os.path.basename(__file__)} {__version__} (running Python {python_version})"
    )

    # matplotlib non-blocking mode, working back-end
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "figure.figsize": [8, 5],
            "font.size": 6,
            "lines.linewidth": 0.5,
            "axes.linewidth": 0.5,
        },
    )
    plt.ion()

    # Start time
    start_time: float = time.time()

    # Select the [Rakic, 1998] model for the metal (Drude-Lorentz or Brendel)
    model: str = "Drude-Lorentz"
    # model: str = "Brendel"
    if model == "Drude-Lorentz":
        rakic_au_model(model="Drude-Lorentz", λ_min=3e-6, λ_max=9e-6, n=100)
        metal_datafile = "Rakic-Au-Drude-Lorentz.xlsx"
    else:
        rakic_au_model(model="Brendel", λ_min=3e-6, λ_max=9e-6, n=100)
        metal_datafile = "Rakic-Au-Brendel.xlsx"

    # Define the material properties for the metal and the insulator
    insulator_datafile = "Kischkat-SiO2.xlsx"
    mats: Materials = Materials(
        insulator_datafile=insulator_datafile,
        insulator_εr_r_model_order=13,
        insulator_εr_i_model_order=13,
        metal_datafile=metal_datafile,
        metal_n_model_order=3,
        metal_κ_model_order=3,
        absorbance_spectrum_sample_count=1000,
        debug=True,
    )

    # Define the reference structure geometry
    geom = Geometry(a=150e-9, b=1.5e-6, Λ=3.6e-6, t_metal=100e-9, t_ins=200e-9, c=0.5)

    # Plot the complex impedance for the reference structure geometry (Zcross)
    plot_z_cross_spectrum(mats=mats, geom=geom)

    # Plot figures from the paper
    fig_2d_absorbers: list = figure_2d(mats=mats, geom=geom)
    save_results_to_file(
        filename="figure_2d.xlsx", absorbers=fig_2d_absorbers, mats=mats, geom=geom
    )
    fig_3a_absorbers: list = figure_3a(mats=mats, geom=geom)
    save_results_to_file(
        filename="figure_3a.xlsx", absorbers=fig_3a_absorbers, mats=mats, geom=geom
    )
    figure_3b(mats=mats, geom=geom)

    # Plot absorbance for "custom" cases
    geom = Geometry(a=150e-9, b=1.5e-6, Λ=3.6e-6, t_metal=50e-9, t_ins=200e-9, c=0.5)
    geometries: np.ndarray = np.asarray(
        [
            [100, 1.3, 3.6],
            [150, 1.5, 3.6],
            [200, 1.7, 3.8],
            [200, 1.9, 3.8],
            [250, 2.1, 4.0],
            [300, 2.3, 4.0],
        ]
    ) * [1e-9, 1e-6, 1e-6]
    custom_absorbers: list = plot_absorbance_spectra(
        mats=mats,
        geom=geom,
        geometries=geometries,
        title="Absorbance (λ) - custom case",
        filename="custom_absorbers.png",
    )
    save_results_to_file(
        filename="custom_absorbers.xlsx",
        absorbers=custom_absorbers,
        mats=mats,
        geom=geom,
    )

    # Show figures
    plt.show()

    # Show running time on console
    print(f"Running time : {time.time() - start_time:.2f} s")

    return None


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
    print("Done!")
