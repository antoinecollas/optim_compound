import autograd.numpy.random as rnd
import os
from pyCovariance.features import identity_euclidean, mean_vector_euclidean

from classification.time_series import Bzh_loader
from classification.MDM_acc_vs_affine_transfo import main


def test_MDM_acc_vs_affine_transfo():
    seed = 0
    rnd.seed(seed)

    bzh_class = Bzh_loader(dataset='unittest', verbose=False)

    features = [
        identity_euclidean(),
        mean_vector_euclidean()
    ]

    main(
        dataset=bzh_class,
        t_list=[0, 0.1, 1],
        mean_transfo=True,
        rotation_transfo=True,
        features=features,
        n_jobs=os.cpu_count(),
        verbose=False
    )

    seed = 0
    rnd.seed(seed)

    bzh_class = Bzh_loader(dataset='unittest', verbose=False)

    features = [
        identity_euclidean(),
        mean_vector_euclidean()
    ]

    main(
        dataset=bzh_class,
        t_list=[0, 0.1, 1],
        mean_transfo=False,
        rotation_transfo=True,
        features=features,
        n_jobs=os.cpu_count(),
        verbose=False
    )
