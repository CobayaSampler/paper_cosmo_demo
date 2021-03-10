from cobaya.likelihoods.base_classes import CMBlikes


class SimulatedLikelihood(CMBlikes):
    """
    Likelihood of a partially-delensed CMB survey dataset.
    """

    Alens_delens = None
    dataset_file = None

    def get_requirements(self):
        req = super().get_requirements()
        req['CAMBdata'] = None
        return req

    def logp(self, **data_params):
        cl_dict = cl_dict_from_camb_results(
            self.provider.get_CAMBdata(), self.Alens_delens)
        return self.log_likelihood(cl_dict, **data_params)


def cl_dict_from_camb_results(camb_results, Alens_delens, lmax=None):
    """
    Returns a dictionary of partially-lensed Cl's from a CAMB results object.
    """
    cls = camb_results.get_partially_lensed_cls(Alens_delens, CMB_unit='muK')
    if lmax is None:
        lmax = len(cls[:, 0])
    return {name: cls[:lmax + 1, index]
            for name, index in {"tt": 0, "ee": 1, "bb": 2, "te": 3}.items()}
