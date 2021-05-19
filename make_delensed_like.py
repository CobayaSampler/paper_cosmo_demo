import os
import sys
import numpy as np
import tempfile
from importlib import import_module
from cobaya.likelihoods.base_classes import make_forecast_cmb_dataset
from cobaya.tools import load_packages_path_from_config_file, get_resolved_class
from likelihood_class import SimulatedLikelihood, cl_dict_from_camb_results
from theory_primordial_Pk import feature_power_spectrum

# Import cobaya-installed CAMB
packages_path = load_packages_path_from_config_file()
camb_path = get_resolved_class("camb").get_path(packages_path)
sys.path.insert(0, camb_path)
camb = import_module("camb", "camb")


lens_potential_accuracy = 5


def make_simulated_dataset(camb_results, Alens_delens, output_root='test_sim',
                           pols=None, output_dir=tempfile.gettempdir(), **kwargs):
    cl_dict = cl_dict_from_camb_results(
        camb_results,  Alens_delens, kwargs.get("lmax"))
    if pols:
        cl_dict = {pol: cl_dict[pol] for pol in pols}
    output_dir = os.path.abspath(output_dir)
    make_forecast_cmb_dataset(cl_dict, output_root=output_root,
                              output_dir=output_dir, **kwargs)
    print('Made ' + os.path.join(output_dir, output_root + '.dataset'))
    like = SimulatedLikelihood(
        {'dataset_file': os.path.join(output_dir, output_root + '.dataset'),
         'Alens_delens': Alens_delens})
    return like


def chi2s(like_dict, cl_dict, experiments):
    chi2s = {}
    for exp, like in like_dict.items():
        this_cl_dict = {pol: cl_dict[pol] for pol in experiments[exp]["pols"]}
        chi2s[exp] = -2 * like.log_likelihood(this_cl_dict)
    return chi2s


def k_accuracy_boost(wavelength):
    """Returns a good IntkAccuracyBoost according to the feature wavelength."""
    return 1 if wavelength > 5e-4 else 2


def l_accuracy_boost(wavelength):
    """Returns a good lSampleBoost according to the feature wavelength."""
    return np.clip(np.ceil(-5.7 * np.log10(wavelength) - 9.2), 3, 7)


def pars_set_feature(pars, As, ns, A, l, c, w, n_samples_wavelength=20):
    """
    Sets a feature primirdial power spectrum in the given CAMBPars instance.
    """
    ks, Pks = feature_power_spectrum(
        As, ns, A, l, c, w, n_samples_wavelength=n_samples_wavelength)
    # Test plot
    # plt.figure()
    # plt.plot(ks, Pks, ".-")
    # plt.semilogx()
    # plt.show()
    initial_power_spectrum = camb.initialpower.SplinedInitialPower(
        effective_ns_for_nonlinear=ns, ks=ks, PK=Pks)
    pars.set_initial_power(initial_power_spectrum)
    pars.Accuracy.IntkAccuracyBoost = k_accuracy_boost(l)
    pars.Accuracy.lSampleBoost = l_accuracy_boost(l)


if __name__ == "__main__":

    output_dir = os.path.join(os.getcwd(), "data")

    # Precision + ranges for the simulated experimental data
    Alens = 0.3  # Same for Planck: ell>2000 should dominate for oscillations
    lmax_SO = 3000
    experiments = {
        "planck_lowl": {"fwhm_arcmin": 5, "fsky": 0.8, "NoiseVar": 4e-5,
                        "ENoiseFac": 4, "lmin": 2, "lmax": 39,
                        "pols": ["tt", "te", "ee"]},
        "planck_highl": {"fwhm_arcmin": 5, "fsky": 0.3, "NoiseVar": 4e-5,
                         "ENoiseFac": 4, "lmin": 40, "lmax": 2500,
                         "pols": ["tt", "te", "ee"]},
        "SO_TEB": {"noise_muK_arcmin_T": 8, "fwhm_arcmin": 1.4, "ENoiseFac": 2,
                   "fsky": 0.4, "lmin": 40, "lmax": lmax_SO,
                   "pols": ["tt", "te", "ee", "bb"]},
        "SO_EB_highl": {"noise_muK_arcmin_T": 8, "fwhm_arcmin": 1.4, "ENoiseFac": 2,
                        "fsky": 0.4, "lmin": lmax_SO + 1, "lmax": 4000,
                        "pols": ["ee", "bb"]}}

    # Prepare ficucial power spectra
    ini = 'planck_2018.ini'
    pars = camb.read_ini(ini)
    As = pars.InitPower.As
    ns = pars.InitPower.ns
    max_lmax = max(exp_defaults.get("lmax", 0) for exp_defaults in experiments.values())
    pars.set_for_lmax(max_lmax, lens_potential_accuracy=lens_potential_accuracy)

    amplitude = 1e-1
    wavelength = 8e-3
    envcentre = 0.2
    envlog10width = 0.1  # in decades

    pars_set_feature(pars, As, ns, amplitude, wavelength, envcentre, envlog10width)
    results_feature = camb.get_results(pars)
    cl_dict_feature = cl_dict_from_camb_results(
        results_feature, Alens_delens=Alens, lmax=max_lmax)
    cl_dict_feature_nodelens = cl_dict_from_camb_results(
        results_feature, Alens_delens=1, lmax=max_lmax)
    ells_feature = np.arange(cl_dict_feature["tt"].shape[0])
    likelihoods_feature = {}
    likelihoods_feature_nodelens = {}
    for exp, exp_defaults in experiments.items():
        likelihoods_feature[exp] = make_simulated_dataset(
            results_feature, Alens_delens=Alens, output_dir=output_dir,
            output_root=exp + "_feature", **exp_defaults)
        likelihoods_feature_nodelens[exp] = make_simulated_dataset(
            results_feature, Alens_delens=1, output_dir=output_dir,
            output_root=exp + "_feature_nodelens", **exp_defaults)

    # Null tests
    chi2s_null_feature = chi2s(
        likelihoods_feature, cl_dict_feature, experiments)
    chi2s_null_feature_nodelens = chi2s(
        likelihoods_feature_nodelens, cl_dict_feature_nodelens, experiments)
    assert abs(sum(sum(chi2 for chi2 in chi2s.values()) for chi2s in
                   [chi2s_null_feature, chi2s_null_feature_nodelens])) < 1e-8, \
        "Feature null test failed"

    # Add the right paths to the input files
    for file_name in ["lensed.yaml", "delensed.yaml"]:
        new_lines = []
        with open(file_name, "r") as this_file:
            for line in this_file:
                new_lines.append(line.replace("/path/to/data", output_dir))
        with open(file_name, "w") as this_file:
            this_file.write("".join(new_lines))
