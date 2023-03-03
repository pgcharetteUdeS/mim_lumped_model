"""mim_lumped_model.py

    Script qui utilise le modèle groupé de l'article "Ultra‐Narrowband Metamaterial
    Absorbers for High Spectral Resolution Infrared Spectroscopy" [Kang, 2019] pour
    faire le calcul de la réponse d'une structure MIM.

    Auteur: Paul Charette

    NB: les paramètres optiques pour l'or (n & k) et le SiO2 (εr) sont calculés
        en fonction de la longueur d'onde par des modèles polynomiaux générés
        à partir de données sur "refractiveindex.info". Les résultats générés
        par le script diffèrent des résulats de la figure 2d de l'article,
        il est probable que ces différences soient dues aux valeurs spécifiques des
        propriétés optiques pour l'or et le SiO2 utilisées.

    Remarques dans la publie:
    1) "When designing an optimized MIM IR absorber with a high spectral selectivity,
        both FWHM and absorption must be considered simultaneously. Since fpeak
        can be independently tuned by b, there exists a set of Λ and a at a given fpeak
        that guarantees a narrow FWHM and a near-unity absorption."
    2) "The upper limit of Λ is set by the wavelength of operation
        (i.e., Λ < Λmax =λpeak/(1 + sinθ), where 0 < θ < 90° is the angle of incidence),
        for Λ exceeding the limit will result in diffraction and the lumped equivalent
        circuit model is no longer valid. Note that this limit is more stringent
        (smaller Λmax) at oblique incidence (θ > 0), making the device more susceptible
        to diffraction. Therefore, to ensure strong absorption over a wide acceptance
        angle, we choose θ = 30° for the upper limit of Λ (e.g., Λmax ≈ 4 μm
        for λpeak = 6 μm).

"""

from itertools import product
from matplotlib import use as plt_use
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from typing import TypedDict

from materials_and_geometry import Geometry, Materials


def c_p(geom: Geometry) -> float:
    """
    Equation 1: periodic coupling capacitance

    Args:
        geom (Geometry): structure geometry

    Returns: Cp

    """

    return (
        (np.pi * constants.epsilon_0)
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

    Returns: Lm

    """

    return 0.5 * constants.mu_0 * geom.t_ox * geom.b / geom.a


def c_m(ω: float, mats: Materials, geom: Geometry) -> complex:
    """
    Equation 3: mutual capacitance

    Args:
        ω (float): radial frequency
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: Cm

    """

    return (
        geom.c
        * constants.epsilon_0
        * mats.ε_ox(λ=constants.c * (2 * np.pi / ω))
        * (geom.b / 2)
        * (geom.a / geom.t_ox)
    )


def l_kc(ω: float, mats: Materials, geom: Geometry) -> float:
    """
    Equation 4: cross nano-structure kinetic inductance

    Args:
        ω (float): radial frequency
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: Lk,c

    """

    return (geom.c_prime * geom.b / (geom.a * mats.δ(ω=ω))) * (
        1 / (constants.epsilon_0 * mats.ω_p**2)
    )


def r_c(ω: float, mats: Materials, geom: Geometry) -> float:
    """
    Equation 5: cross nano-structure resistance

    Args:
        ω (float): radial frequency
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: Rc

    """

    return (geom.c_prime * geom.b / (geom.a * mats.δ(ω=ω))) * (1 / mats.σ)


def l_kg(ω: float, mats: Materials) -> float:
    """
    Equation 6: ground plane kinetic inductance

    Args:
        ω (float): radial frequency
        mats (Materials): material properties

    Returns: Lk,g

    """

    return (2 / mats.δ(ω=ω)) * (1 / (constants.epsilon_0 * mats.ω_p**2))


def r_g(ω: float, mats: Materials) -> float:
    """
    Equation 7: ground plane resistance

    Args:
        ω (float): radial frequency
        mats (Materials): material properties

    Returns: Rg

    """

    return (2 / mats.δ(ω=ω)) * (1 / mats.σ)


def z_cross(ω: float, mats: Materials, geom: Geometry) -> complex:
    """
    Equation S9: complex total impedance

    Args:
        ω (float): radial frequency
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: Zcross

    """

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

    return z_e + z_m


def absorbance_spectrum(λ: float, mats: Materials, geom: Geometry) -> float:
    """
    Absorbance as a function of wavelength

    Args:
        λ (float): wavelength (m)
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: Absorbance spectrum

    """

    z0 = np.sqrt(constants.mu_0 / constants.epsilon_0)
    ω: float = 2 * np.pi * (constants.c / λ)
    z_cross_val: complex = z_cross(ω=ω, mats=mats, geom=geom)
    r_in: float = np.abs((z_cross_val - z0) / (z_cross_val + z0)) ** 2

    return 1 - r_in


class FilterResponseMetrics(TypedDict):
    """
    Typed dictionary data type for filter_response_metrics() return values
    """

    λ_peak: float
    fwhm: float
    q: float
    absorbance: np.ndarray


def filter_response_metrics(
    λs: np.ndarray,
    mats: Materials,
    geom: Geometry,
    a: float,
    b: float,
    Λ: float,
) -> FilterResponseMetrics:
    """
    Filter response metrics λpeak, FWHM, and Q as a function
    of structure geometry parameters a, b, and Λ

    Args:
        λs (np.ndarray): array of wavelengths (m)
        mats (Materials): material properties
        geom (Geometry): reference structure geometry
        a (float): cross arm width (m)
        b (float): cross arm length (m)
        Λ (float): cross pattern period (m)

    Returns: λ_peak, FWHM, Q, array of absorbance

    """

    # Load geometry parameters into the reference geometry object
    geom.a = a
    geom.b = b
    geom.Λ = Λ

    # Absorbance as a function of wavelength
    absorbance: np.ndarray = np.asarray(
        [absorbance_spectrum(λ=λ, mats=mats, geom=geom) for λ in λs]
    )

    # Find absorbance peak wavelength
    λ_peak_index = int(absorbance.argmax())
    λ_peak: float = λs[λ_peak_index]

    # FWHM
    i_left: int = np.absolute(absorbance[:λ_peak_index] - 0.5).argmin()
    i_right: int = np.absolute(absorbance[λ_peak_index:] - 0.5).argmin()
    fwhm: float = λs[λ_peak_index + i_right] - λs[i_left]

    # Q
    q: float = λ_peak / fwhm

    # Return filter metrics and absorbance spectrum
    return {"λ_peak": λ_peak, "fwhm": fwhm, "q": q, "absorbance": absorbance}


def figure_2d(λs: np.ndarray, mats: Materials, geom: Geometry):
    """
    Plot Figure 2d from the paper

    Args:
        λs (np.ndarray): arrays of wavelengths for the analysis
        mats (Materials): material properties
        geom (Geometry): reference structure geometry

    Returns: None

    """

    # Arrays of cross widths (a), lengths (b) and periods (Λ) for the MIM structures
    a_array: np.ndarray = np.asarray([150, 200, 300, 350]) * 1e-9
    b_array: np.ndarray = np.asarray([1.5, 1.7, 1.9, 2.1]) * 1e-6
    Λ_array: np.ndarray = np.asarray([3.6, 3.8, 4.0, 4.2]) * 1e-6

    # Loop to plot absorbance as a function of wavelength for the MIM structures
    fig, ax = plt.subplots()
    for a, b, Λ in zip(a_array, b_array, Λ_array):
        λ_peak, fwhm, q, absorbance = filter_response_metrics(
            λs=λs,
            mats=mats,
            geom=geom,
            a=a,
            b=b,
            Λ=Λ,
        ).values()
        ax.plot(
            λs * 1e6,
            absorbance,
            label=rf"λ$_{{peak}}$={λ_peak*1e6:.2f} μm, "
            f"FWHM={fwhm*1e9:.0f} nm, "
            f"Q={q:.1e}\n"
            f"b={b*1e6:.1f} μm, a={a*1e9:.0f} nm, Λ={Λ*1e6:.1f} μm",
        )
    ax.set(
        title="Figure 2d : Absorbance(λ)",
        xlabel="Wavelength (μm)",
        ylabel="Absorbance",
        ylim=([0, 1]),
    )
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

    return None


def figure_3b(λs: np.ndarray, mats: Materials, geom: Geometry):
    """
    Plot Figure 3b from the paper, as well as a 2D map of FWHM(Λ, a)

    Args:
        λs (np.ndarray): arrays of wavelengths for the analysis
        mats (Materials): material properties
        geom (Geometry): reference structure geometry

    Returns: None

    """

    # Arrays of cross widths (a), lengths (b) and periods (Λ) for the MIM structures
    n: int = 10
    a_array: np.ndarray = np.linspace(150, 350, n) * 1e-9
    b_array: np.ndarray = np.linspace(1.5, 2.4, n) * 1e-6
    Λ_array: np.ndarray = np.linspace(3.6, 4.2, n) * 1e-6

    # Figure 3b
    a_fixed: float = 200e-9
    Λ_fixed: float = 4e-6
    λ_peak_array = np.asarray(
        [
            filter_response_metrics(
                λs=λs,
                mats=mats,
                geom=geom,
                a=a_fixed,
                b=b,
                Λ=Λ_fixed,
            ).get("λ_peak")
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
    λ_peak_fixed: float = filter_response_metrics(
        λs=λs,
        mats=mats,
        geom=geom,
        a=a_fixed,
        b=b_fixed,
        Λ=Λ_fixed,
    ).get("λ_peak")
    fwhm_array = np.asarray(
        [
            filter_response_metrics(
                λs=λs,
                mats=mats,
                geom=geom,
                a=a,
                b=b_fixed,
                Λ=Λ,
            ).get("fwhm")
            for Λ, a in product(Λ_array, a_array)
        ]
    ).reshape((len(Λ_array), len(a_array)))
    fig, ax = plt.subplots()
    im = ax.imshow(
        np.flipud(fwhm_array) * 1e6,
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
    fig.colorbar(im, label="FWHM (μm)")

    return None


def main():
    """
    Main calling function

    Returns: None

    """

    # matplotlib non-blocking mode, working back-end
    plt_use("TkAgg")
    plt.ion()

    # Define metal and oxyde materials properties in a Materials class object
    mats: Materials = Materials(ω_p=2 * np.pi * 2.183e15, τ=12.4e-15)

    # Define reference structure geometry in a Geometry class object
    geom = Geometry(a=150e-9, b=1.5e-6, Λ=3.6e-9, t_metal=100e-9, t_ox=200e-9, c=0.4)

    # Wavelength spectrum analysis domain
    λs: np.ndarray = np.linspace(4e-6, 7e-6, 1000)

    # Figure 2d from the paper
    figure_2d(λs=λs, mats=mats, geom=geom)

    # Figure 3b from the paper
    figure_3b(λs=λs, mats=mats, geom=geom)

    return None


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
    print("Done!")
