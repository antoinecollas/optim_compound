import autograd.numpy.random as rnd
import os

from classification.time_series import Bzh_loader
from classification.MDM_tuning import main


def test_MDM_tuning():
    seed = 0
    rnd.seed(seed)

    bzh_class = Bzh_loader(dataset='unittest_small', verbose=False)

    ALPHA_LIST = [0]

    REG_DICT = {
        'L2': [1e-8, 1e-6]
    }

    MIN_GRAD_NORM_MEAN = 1e-2

    main(
        dataset=bzh_class,
        alpha_list=ALPHA_LIST,
        symmetrize_div=False,
        reg_dict=REG_DICT,
        min_grad_norm_mean=MIN_GRAD_NORM_MEAN,
        n_jobs=os.cpu_count(),
        verbose=False
    )
