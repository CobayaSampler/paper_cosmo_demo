import numpy as np
from cobaya.theory import Theory


def feature_envelope(k, c, w):
    """
    Returns the value of the envelope at k. The envelope functional form is

        env(k) = exp(-0.5 (log10(k/c)/w)**2) * sin(2 pi k/l)
    """
    return np.exp(-0.5 * (np.log10(k / c) / w) ** 2)


def feature_power_spectrum(As, ns, A, l, c, w,
                           kmin=1e-6, kmax=10, # generous, for transfer integrals
                           k_pivot=0.05, n_samples_wavelength=20):
    """
    Creates the primordial scalar power spectrum as a power law plus an oscillatory
    feature of given amplitude A and wavelength l, centred at c with a lognormal envelope
    of log10-width w, as

        Delta P/P = A * exp(-0.5 (log10(k/c)/w)**2) * sin(2 pi k/l)

    The characteristic delta_k is determined by the number of samples per oscillation
    n_samples_wavalength (default: 20).

    Returns a sample of k, P(k)
    """
    # Ensure thin enough sampling at low-k
    delta_k = min(0.0005, l / n_samples_wavelength)
    ks = np.arange(kmin, kmax, delta_k)
    power_law = lambda k: As * (k / k_pivot) ** (ns - 1)
    DeltaP_over_P = lambda k: (
            A * feature_envelope(k, c, w) * np.sin(2 * np.pi * k / l))
    Pks = power_law(ks) * (1 + DeltaP_over_P(ks))
    return ks, Pks


class FeaturePrimordialPk(Theory):
    """
    Theory class producing a slow-roll-like power spectrum with an enveloped,
    linearly-oscillatory feture on top.
    """

    params = {"As": None, "ns": None,
              "amplitude": None, "wavelength": None, "centre": None, "logwidth": None}

    n_samples_wavelength = 20
    k_pivot = 0.05

    def calculate(self, state, want_derived=True, **params_values_dict):
        As, ns, amplitude, wavelength, centre, logwidth = \
            [params_values_dict[p] for p in
             ["As", "ns", "amplitude", "wavelength", "centre", "logwidth"]]
        ks, Pks = feature_power_spectrum(
            As, ns, amplitude, wavelength, centre, logwidth, kmin=1e-6, kmax=10,
            k_pivot=self.k_pivot, n_samples_wavelength=self.n_samples_wavelength)
        state['primordial_scalar_pk'] = {'k': ks, 'Pk': Pks, 'log_regular': False}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']


def logprior_high_k(A, c, w, k_high=0.25, A_min=5e-3):
    """
    Returns -inf whenever the feature acts at too high k's only, i.e. such that the
    product of amplituce and evenlope at `k_high` is smaller than `A_min`, given that the
    envelope is centred at `k > k_high`.
    """
    if c < k_high:
        return 0
    return 0 if A * feature_envelope(k_high, c, w) > A_min else -np.inf
