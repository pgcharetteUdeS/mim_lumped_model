"""materials_and_geometry.py

"""
__all__ = ["Geometry", "Materials"]

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy import constants


class Geometry:
    """
    MIM structure geometry

    self.a (float) : cross arm width (m)
    self.b (float) : cross arm length (m)
    self.Λ (float) : cross pattern period (m)
    self.t_metal (float) : cross metal film thickness (m)
    self.t_ox (float) : oxyde spacer film thickness (m)
    self.c (float) : constant chosen to take the fringe effect of the capacitance
                     and nonuniform electric field distribution into consideration

    """

    def __init__(
        self, a: float, b: float, Λ: float, t_metal: float, t_ox: float, c: float
    ):
        self.a = a
        self.b = b
        self.Λ = Λ
        self.t_metal = t_metal
        self.t_ox = t_ox
        self.c = c
        self.c_prime = 1 - c


class Materials:
    """
    Metal and oxyde material properties

    self.ω_p (float): metal plasma frequency (rads/s)
    self.τ (float): metal relaxation time (s)
    self.σ (float): metal DC conductivity (ohms * m)
    self.debug (bool): enable/disable plotting of modeling results for model validation
    self.εr_r_ox, self.εr_i_ox (np.ndarray, np.ndarray): polynomial model coefficients
                                                   for real/imaginary components
                                                   of oxyde relative permittivity εr(λ)
    self.n_metal, self.κ (np.ndarray, np.ndarray): polynomial model coefficients
                                                   for real/imaginary components
                                                   of metal optical index n + iκ

    """

    def __init__(self, ω_p: float, τ: float, debug: bool = False):
        # Initialize class instance variables
        self.ω_p: float = ω_p
        self.τ: float = τ
        self.σ: float = constants.epsilon_0 * self.ω_p**2 * self.τ
        self.debug: bool = debug

        # Load polynomial models for metal and oxyde optical parameters
        self.εr_r_ox, self.εr_i_ox = self.fit_oxyde_permittivity_model()
        self.n_metal, self.κ = self.fit_metal_optical_index_model()

    def ε_ox(self, λ: float) -> complex:
        """
        oxyde complex relative permittivity at wavelength λ (polynomial model)

        Args:
            λ (float): wavelength (m)

        Returns: εr

        """

        return poly.polyval(λ * 1e6, self.εr_r_ox) + 1j * poly.polyval(
            λ * 1e6, self.εr_i_ox
        )

    def δ(self, ω: float) -> float:
        """

        Equation 8: metal skin depth at radial frequency ω (polynomial model)

        Args:
            ω (float): radial frequency (Hz)

        Returns: δ (metal skin depth)

        """

        λ: float = constants.c * (2 * np.pi / ω)
        κ: float = float(poly.polyval(λ * 1e6, self.κ))

        return λ / (2 * np.pi * κ)

    def fit_metal_optical_index_model(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit polynomial model to Au data from refractiveindex.info, [Ciesielski, 2018]

        Returns: polynomial models for Au n & κ

        """

        # Data (λ, n, k)
        gold_optical_index: np.ndarray = np.asarray(
            [
                [2.5827, 1.284, 17.245],
                [2.6899, 1.388, 17.966],
                [2.8064, 1.506, 18.747],
                [2.9333, 1.640, 19.597],
                [3.0724, 1.794, 20.523],
                [3.2252, 1.970, 21.538],
                [3.3941, 2.174, 22.654],
                [3.5816, 2.412, 23.887],
                [3.7911, 2.692, 25.258],
                [4.0265, 3.023, 26.788],
                [4.2932, 3.420, 28.509],
                [4.5977, 3.900, 30.456],
                [4.9486, 4.487, 32.678],
                [5.3576, 5.215, 35.235],
                [5.8403, 6.134, 38.205],
                [6.4185, 7.313, 41.695],
                [7.1239, 8.857, 45.845],
                [8.0033, 10.927, 50.849],
                [9.1305, 13.781, 56.982],
                [10.627, 17.841, 64.631],
            ]
        )

        # Fit polynomial models (order determined by trial & error)
        n_poly, n_stats = poly.polyfit(
            x=gold_optical_index[:, 0], y=gold_optical_index[:, 1], deg=3, full=True
        )
        κ_poly, κ_stats = poly.polyfit(
            x=gold_optical_index[:, 0], y=gold_optical_index[:, 2], deg=4, full=True
        )

        # If required, plot data and model estimates for validation of model oder
        if self.debug:
            fig, [ax0, ax1] = plt.subplots(2)
            fig.suptitle("Model fits to n&k data for au")
            ax0.plot(gold_optical_index[:, 0], gold_optical_index[:, 1])
            ax0.plot(
                gold_optical_index[:, 0],
                poly.polyval(gold_optical_index[:, 0], n_poly),
                "--",
            )
            ax0.set(title=f"n (erms = {n_stats[0][0]:.2e})")
            ax0.grid()
            ax1.plot(gold_optical_index[:, 0], gold_optical_index[:, 2])
            ax1.plot(
                gold_optical_index[:, 0],
                poly.polyval(gold_optical_index[:, 0], κ_poly),
                "--",
            )
            ax1.set(title=f"κ (erms = {κ_stats[0][0]:.2e})", xlabel="Wavelength (μm)")
            ax1.grid()
            plt.tight_layout()
            plt.savefig("Au_data_and_model_fit.png")
            plt.show()

        return n_poly, κ_poly

    def fit_oxyde_permittivity_model(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit polynomial model to SiO2 data from refractiveindex.info, [Kischkat, 2012]

        Returns: polynomial models for SiO2 complex permittivity components

        """

        # Data (λ, n, k)
        sio2_optical_index: np.ndarray = np.asarray(
            [
                [2.50000, 1.43030, 0.00008],
                [2.53036, 1.42977, 0.00008],
                [2.56148, 1.42921, 0.00008],
                [2.59336, 1.42864, 0.00009],
                [2.62605, 1.42804, 0.00009],
                [2.65957, 1.42741, 0.00009],
                [2.69397, 1.42676, 0.00010],
                [2.72926, 1.42608, 0.00010],
                [2.76549, 1.42537, 0.00011],
                [2.80269, 1.42462, 0.00011],
                [2.84091, 1.42384, 0.00011],
                [2.88018, 1.42303, 0.00012],
                [2.92056, 1.42218, 0.00013],
                [2.96209, 1.42128, 0.00013],
                [3.00481, 1.42034, 0.00014],
                [3.04878, 1.41935, 0.00014],
                [3.09406, 1.41832, 0.00015],
                [3.14070, 1.41722, 0.00016],
                [3.18878, 1.41607, 0.00017],
                [3.23834, 1.41486, 0.00017],
                [3.28947, 1.41358, 0.00018],
                [3.34225, 1.41223, 0.00019],
                [3.39674, 1.41079, 0.00020],
                [3.45304, 1.40927, 0.00021],
                [3.51124, 1.40766, 0.00023],
                [3.57143, 1.40595, 0.00024],
                [3.63372, 1.40413, 0.00025],
                [3.69822, 1.40219, 0.00027],
                [3.76506, 1.40012, 0.00028],
                [3.83436, 1.39790, 0.00030],
                [3.90625, 1.39552, 0.00032],
                [3.98089, 1.39297, 0.00034],
                [4.05844, 1.39023, 0.00036],
                [4.13907, 1.38727, 0.00039],
                [4.22297, 1.38408, 0.00042],
                [4.31034, 1.38061, 0.00045],
                [4.40141, 1.37684, 0.00048],
                [4.49640, 1.37274, 0.00052],
                [4.59559, 1.36825, 0.00056],
                [4.69925, 1.36331, 0.00060],
                [4.80769, 1.35788, 0.00066],
                [4.92126, 1.35186, 0.00071],
                [5.00000, 1.34748, 0.00076],
                [5.01002, 1.34691, 0.00076],
                [5.02008, 1.34633, 0.00077],
                [5.03018, 1.34575, 0.00078],
                [5.04032, 1.34517, 0.00078],
                [5.05051, 1.34457, 0.00079],
                [5.06073, 1.34398, 0.00079],
                [5.07099, 1.34337, 0.00080],
                [5.08130, 1.34276, 0.00081],
                [5.09165, 1.34215, 0.00081],
                [5.10204, 1.34153, 0.00082],
                [5.11247, 1.34090, 0.00082],
                [5.12295, 1.34027, 0.00083],
                [5.13347, 1.33963, 0.00084],
                [5.14403, 1.33899, 0.00084],
                [5.15464, 1.33833, 0.00085],
                [5.16529, 1.33768, 0.00086],
                [5.17598, 1.33701, 0.00086],
                [5.18672, 1.33634, 0.00087],
                [5.19751, 1.33566, 0.00088],
                [5.20833, 1.33498, 0.00089],
                [5.21921, 1.33429, 0.00089],
                [5.23013, 1.33359, 0.00090],
                [5.24109, 1.33288, 0.00091],
                [5.25210, 1.33217, 0.00091],
                [5.26316, 1.33145, 0.00092],
                [5.27426, 1.33072, 0.00093],
                [5.28541, 1.32999, 0.00094],
                [5.29661, 1.32924, 0.00095],
                [5.30786, 1.32849, 0.00095],
                [5.31915, 1.32774, 0.00096],
                [5.33049, 1.32697, 0.00097],
                [5.34188, 1.32619, 0.00098],
                [5.35332, 1.32541, 0.00099],
                [5.36481, 1.32462, 0.00099],
                [5.37634, 1.32382, 0.00100],
                [5.38793, 1.32301, 0.00101],
                [5.39957, 1.32219, 0.00102],
                [5.41126, 1.32137, 0.00103],
                [5.42299, 1.32053, 0.00104],
                [5.43478, 1.31969, 0.00105],
                [5.44662, 1.31883, 0.00106],
                [5.45852, 1.31797, 0.00107],
                [5.47046, 1.31710, 0.00108],
                [5.48246, 1.31621, 0.00109],
                [5.49451, 1.31532, 0.00110],
                [5.50661, 1.31442, 0.00111],
                [5.51876, 1.31350, 0.00112],
                [5.53097, 1.31258, 0.00113],
                [5.54324, 1.31164, 0.00114],
                [5.55556, 1.31069, 0.00115],
                [5.56793, 1.30974, 0.00116],
                [5.58036, 1.30877, 0.00117],
                [5.59284, 1.30779, 0.00118],
                [5.60538, 1.30679, 0.00119],
                [5.61798, 1.30579, 0.00120],
                [5.63063, 1.30477, 0.00121],
                [5.64334, 1.30374, 0.00122],
                [5.65611, 1.30270, 0.00124],
                [5.66893, 1.30165, 0.00125],
                [5.68182, 1.30058, 0.00126],
                [5.69476, 1.29950, 0.00127],
                [5.70776, 1.29840, 0.00129],
                [5.72082, 1.29730, 0.00130],
                [5.73394, 1.29617, 0.00131],
                [5.74713, 1.29504, 0.00132],
                [5.76037, 1.29388, 0.00134],
                [5.77367, 1.29272, 0.00135],
                [5.78704, 1.29154, 0.00137],
                [5.80046, 1.29034, 0.00138],
                [5.81395, 1.28912, 0.00139],
                [5.82751, 1.28789, 0.00141],
                [5.84112, 1.28665, 0.00142],
                [5.85480, 1.28538, 0.00144],
                [5.86854, 1.28410, 0.00145],
                [5.88235, 1.28281, 0.00147],
                [5.89623, 1.28149, 0.00149],
                [5.91017, 1.28016, 0.00150],
                [5.92417, 1.27880, 0.00152],
                [5.93824, 1.27743, 0.00153],
                [5.95238, 1.27604, 0.00155],
                [5.96659, 1.27463, 0.00157],
                [5.98086, 1.27319, 0.00159],
                [5.99520, 1.27174, 0.00160],
                [6.00962, 1.27027, 0.00162],
                [6.02410, 1.26877, 0.00164],
                [6.03865, 1.26725, 0.00166],
                [6.05327, 1.26571, 0.00168],
                [6.06796, 1.26415, 0.00170],
                [6.08273, 1.26256, 0.00172],
                [6.09756, 1.26095, 0.00174],
                [6.11247, 1.25931, 0.00176],
                [6.12745, 1.25764, 0.00178],
                [6.14251, 1.25595, 0.00181],
                [6.15764, 1.25424, 0.00183],
                [6.17284, 1.25249, 0.00185],
                [6.18812, 1.25072, 0.00187],
                [6.20347, 1.24892, 0.00190],
                [6.21891, 1.24709, 0.00192],
                [6.23441, 1.24523, 0.00195],
                [6.25000, 1.24334, 0.00197],
                [6.26566, 1.24141, 0.00200],
                [6.28141, 1.23946, 0.00203],
                [6.29723, 1.23747, 0.00205],
                [6.31313, 1.23544, 0.00208],
                [6.32911, 1.23338, 0.00211],
                [6.34518, 1.23128, 0.00214],
                [6.36132, 1.22915, 0.00217],
                [6.37755, 1.22698, 0.00220],
                [6.39386, 1.22476, 0.00223],
                [6.41026, 1.22251, 0.00227],
                [6.42674, 1.22022, 0.00230],
                [6.44330, 1.21788, 0.00233],
                [6.45995, 1.21549, 0.00237],
                [6.47668, 1.21307, 0.00241],
                [6.49351, 1.21059, 0.00244],
                [6.51042, 1.20807, 0.00248],
                [6.52742, 1.20549, 0.00252],
                [6.54450, 1.20286, 0.00256],
                [6.56168, 1.20019, 0.00261],
                [6.57895, 1.19745, 0.00265],
                [6.59631, 1.19466, 0.00269],
                [6.61376, 1.19181, 0.00274],
                [6.63130, 1.18890, 0.00279],
                [6.64894, 1.18593, 0.00284],
                [6.66667, 1.18289, 0.00289],
                [6.68449, 1.17978, 0.00295],
                [6.70241, 1.17661, 0.00300],
                [6.72043, 1.17336, 0.00306],
                [6.73854, 1.17004, 0.00312],
                [6.75676, 1.16664, 0.00318],
                [6.77507, 1.16316, 0.00325],
                [6.79348, 1.15960, 0.00331],
                [6.81199, 1.15595, 0.00339],
                [6.83060, 1.15221, 0.00346],
                [6.84932, 1.14838, 0.00354],
                [6.86813, 1.14445, 0.00362],
                [6.88705, 1.14042, 0.00370],
                [6.90608, 1.13628, 0.00379],
                [6.92521, 1.13204, 0.00389],
                [6.94444, 1.12767, 0.00398],
                [6.96379, 1.12319, 0.00409],
                [6.98324, 1.11858, 0.00420],
                [7.00280, 1.11384, 0.00432],
                [7.02247, 1.10896, 0.00444],
                [7.04225, 1.10394, 0.00457],
                [7.06215, 1.09877, 0.00471],
                [7.08215, 1.09343, 0.00486],
                [7.10227, 1.08793, 0.00502],
                [7.12251, 1.08225, 0.00520],
                [7.14286, 1.07639, 0.00538],
                [7.16332, 1.07032, 0.00558],
                [7.18391, 1.06406, 0.00580],
                [7.20461, 1.05757, 0.00603],
                [7.22543, 1.05085, 0.00629],
                [7.24638, 1.04388, 0.00657],
                [7.26744, 1.03665, 0.00688],
                [7.28863, 1.02913, 0.00722],
                [7.30994, 1.02132, 0.00759],
                [7.33138, 1.01319, 0.00800],
                [7.35294, 1.00472, 0.00846],
                [7.37463, 0.99589, 0.00898],
                [7.39645, 0.98666, 0.00956],
                [7.41840, 0.97700, 0.01022],
                [7.44048, 0.96689, 0.01096],
                [7.46269, 0.95629, 0.01181],
                [7.48503, 0.94515, 0.01278],
                [7.50751, 0.93343, 0.01390],
                [7.53012, 0.92108, 0.01520],
                [7.55287, 0.90804, 0.01672],
                [7.57576, 0.89424, 0.01849],
                [7.59878, 0.87963, 0.02058],
                [7.62195, 0.86413, 0.02307],
                [7.64526, 0.84764, 0.02603],
                [7.66871, 0.83010, 0.02959],
                [7.69231, 0.81140, 0.03388],
                [7.71605, 0.79147, 0.03908],
                [7.73994, 0.77021, 0.04543],
                [7.76398, 0.74758, 0.05319],
                [7.78816, 0.72356, 0.06272],
                [7.81250, 0.69819, 0.07443],
                [7.83699, 0.67163, 0.08881],
                [7.86164, 0.64421, 0.10639],
                [7.88644, 0.61646, 0.12770],
                [7.91139, 0.58921, 0.15315],
                [7.93651, 0.56351, 0.18289],
                [7.96178, 0.54054, 0.21658],
                [7.98722, 0.52132, 0.25340],
            ]
        )
        εr_r: np.ndarray = sio2_optical_index[:, 1] ** 2 - sio2_optical_index[:, 2] ** 2
        εr_i: np.ndarray = 2 * sio2_optical_index[:, 1] * sio2_optical_index[:, 2]

        # Fit polynomial models (order determined by trial & error)
        εr_r_poly, εr_r_stats = poly.polyfit(
            x=sio2_optical_index[:, 0], y=εr_r, deg=9, full=True
        )
        εr_i_poly, εr_i_stats = poly.polyfit(
            x=sio2_optical_index[:, 0], y=εr_i, deg=13, full=True
        )

        # If required, plot data and model estimates for validation of model oder
        if self.debug:
            εr_r_modeled: np.ndarray = poly.polyval(sio2_optical_index[:, 0], εr_r_poly)
            εr_i_modeled: np.ndarray = poly.polyval(sio2_optical_index[:, 0], εr_i_poly)
            common_term: np.ndarray = np.sqrt(εr_r_modeled**2 + εr_i_modeled**2)
            n_modeled: np.ndarray = np.sqrt(0.5 * (common_term + εr_r_modeled))
            κ_modeled: np.ndarray = np.sqrt(0.5 * (common_term - εr_r_modeled))

            fig, [ax0, ax1] = plt.subplots(2)
            fig.suptitle("Model fits to n&k data for SiO2")
            ax0.plot(sio2_optical_index[:, 0], sio2_optical_index[:, 1])
            ax0.plot(sio2_optical_index[:, 0], n_modeled, "--")
            ax0.set(title=f"n (erms = {εr_r_stats[0][0]:.2e})")
            ax0.grid()
            ax1.plot(sio2_optical_index[:, 0], sio2_optical_index[:, 2])
            ax1.plot(sio2_optical_index[:, 0], κ_modeled, "--")
            ax1.set(
                title=f"κ (erms = {εr_i_stats[0][0]:.2e})", xlabel="Wavelength (μm)"
            )
            ax1.grid()
            plt.tight_layout()
            plt.savefig("SiO2_data_and_model_fit.png")
            plt.show()

        return εr_r_poly, εr_i_poly
