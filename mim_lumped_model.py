"""mim_lumped_model.py

    Script qui génère la figure 4d de l'article "Ultra‐Narrowband Metamaterial
    Absorbers for High Spectral Resolution Infrared Spectroscopy" [Kang, 2019].

    Auteur: Paul Charette

    NB: les paramètres optiques pour l'or (n & k) et le SiO2 (εr) sont calculés
        en fonction de la longueur d'onde par des modèles polynomiaux générés
        à partir de données sur "refractiveindex.info". Les résultats générés
        par le script diffèrent des résulats de la figure 4d de l'article,
        il est probable que ces différences soient dues aux valeurs spécifiques des
        propriétés optiques pour l'or et le SiO2 utilisées.

"""

from matplotlib import use as plt_use
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

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
            2 * (geom.Λ - geom.b) / geom.t_au
            + np.sqrt((2 * (geom.Λ - geom.b) / geom.t_au) ** 2 - 1)
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

    return (geom.c_prime * geom.b / (geom.a * mats.δ_au(ω=ω))) * (
        1 / (constants.epsilon_0 * mats.ω_p_au**2)
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

    return (geom.c_prime * geom.b / (geom.a * mats.δ_au(ω=ω))) * (1 / mats.σ_au)


def l_kg(ω: float, mats: Materials) -> float:
    """
    Equation 6: ground plane kinetic inductance

    Args:
        ω (float): radial frequency
        mats (Materials): material properties

    Returns: Lk,g

    """

    return (2 / mats.δ_au(ω=ω)) * (1 / (constants.epsilon_0 * mats.ω_p_au**2))


def r_g(ω: float, mats: Materials) -> float:
    """
    Equation 7: ground plane resistance

    Args:
        ω (float): radial frequency
        mats (Materials): material properties

    Returns: Rg

    """

    return (2 / mats.δ_au(ω=ω)) * (1 / mats.σ_au)


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


def absorbance(λ: float, mats: Materials, geom: Geometry) -> float:
    """
    Absorbance

    Args:
        λ (float): wavelength (m)
        mats (Materials): material properties
        geom (Geometry): structure geometry

    Returns: A

    """

    z0 = np.sqrt(constants.mu_0 / constants.epsilon_0)
    ω: float = 2 * np.pi * (constants.c / λ)
    z_cross_val: complex = z_cross(ω=ω, mats=mats, geom=geom)
    r_in: float = np.abs((z_cross_val - z0) / (z_cross_val + z0)) ** 2

    return 1 - r_in


def main():
    """

    Returns: None

    """

    # matplotlib non-blocking mode, working back-end
    plt_use("TkAgg")
    plt.ion()

    # Define material properties
    mats: Materials = Materials()

    # Wavelength solution domain
    λs: np.ndarray = np.linspace(4e-6, 7e-6, 1000)

    # Define structure geometries
    t_au: float = 100e-9
    t_ox: float = 200e-9
    c: float = 0.4
    cross_arm_widths_a: np.ndarray = np.asarray([150, 200, 300, 350]) * 1e-9
    cross_arm_lengths_b: np.ndarray = np.asarray([1.5, 1.7, 1.9, 2.1]) * 1e-6
    cross_arm_periods_Λ: np.ndarray = np.asarray([3.6, 3.8, 4.0, 4.2]) * 1e-6

    # Loop to plot absorbance as a function of wavelength for the different geometries
    fig, ax = plt.subplots()
    for a, b, Λ in zip(cross_arm_widths_a, cross_arm_lengths_b, cross_arm_periods_Λ):
        geom = Geometry(a=a, b=b, Λ=Λ, t_au=t_au, t_ox=t_ox, c=c)
        absorbances: np.ndarray = np.asarray(
            [absorbance(λ=λ, mats=mats, geom=geom) for λ in λs]
        )
        λ_peak_um = λs[np.argmax(absorbances)] * 1e6
        ax.plot(
            λs * 1e6,
            absorbances,
            label=rf"λ$_{{peak}}$={λ_peak_um:.2f} μm"
            f" (b={b*1e6:.1f} μm, a={a*1e9:.0f} nm, Λ={Λ*1e6:.1f} μm)",
        )
    ax.set(
        title="Absorbance vs wavelength",
        xlabel="Wavelength (μm)",
        ylabel="Absorbance",
        ylim=([0, 1]),
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
    print("Done!")
