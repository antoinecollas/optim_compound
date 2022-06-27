import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import numpy.testing as np_testing
import os
import tikzplotlib
from tqdm import tqdm
from scipy.linalg import expm

from pyCovariance.classification import MDM
from pyCovariance.evaluation import create_directory
from pyCovariance.features import\
        covariance_euclidean,\
        covariance_div_alpha,\
        location_covariance_div_alpha,\
        location_covariance_texture_constrained_texture_div_alpha,\
        identity_euclidean,\
        mean_vector_euclidean

from classification.time_series import Bzh_loader, compute_performance_metrics


def main(
    dataset,
    t_list,
    mean_transfo,
    rotation_transfo,
    features,
    n_jobs,
    verbose=True
):
    matplotlib.use('Agg')

    # folder path to save files
    folder_name = 'MDM_acc_vs_affine_transfo_'
    folder_name += dataset.dataset + '_' + dataset.level
    folder_name = os.path.join('classification', folder_name)
    folder = create_directory(folder_name)

    # load data
    X_train_raw, y_train = dataset.get_data_train()
    X_test_raw, y_test = dataset.get_data_test()
    p, N, K = dataset.get_info()

    if verbose:
        print('Dataset:', dataset.dataset)
        print('p:', p)
        print('Length of time series:', N)
        print('Number of classes:', K)
        print('X_train_raw.shape:', X_train_raw.shape)
        print('X_test_raw.shape:', X_test_raw.shape)
        print()

    # store all results
    performance_metrics = dict()

    # MDM
    features_str = list()
    if verbose:
        iterator = enumerate(tqdm(t_list))
    else:
        iterator = enumerate(t_list)

    # generate tranformations parameters
    MU_0_NORM = 5
    mu_0 = rnd.normal(size=(p, 1))
    mu_0 *= MU_0_NORM / la.norm(mu_0)
    tmp = rnd.normal(size=(p, p))
    M = (tmp - tmp.T) / 2

    for j, t in iterator:
        # train
        X_train = deepcopy(X_train_raw)

        # test
        if mean_transfo:
            mu = t * mu_0
        else:
            mu = np.zeros_like(mu_0)
        if rotation_transfo:
            Q = expm(t * M)
        else:
            Q = np.eye(p)
        np_testing.assert_allclose(Q.T @ Q, np.eye(p), rtol=1e-7, atol=1e-7)
        X_test = (Q.T @ deepcopy(X_test_raw)) + mu

        for i, feature in enumerate(features):
            # MDM
            clf = MDM(
                feature=feature,
                n_jobs=n_jobs,
                verbose=False
            )
            y_pred = clf.fit_predict(X_train, y_train)
            compute_performance_metrics(
                y_train, y_pred, classnames=dataset.classname, verbose=False)

            if len(features_str) < len(features):
                features_str.append(str(clf.feature))

            # evaluation
            y_pred = clf.predict(X_test)
            new_performance_metrics = compute_performance_metrics(
                y_test, y_pred, classnames=dataset.classname, verbose=False)
            for key in new_performance_metrics:
                if not(key in performance_metrics):
                    init = np.zeros((len(features), len(t_list)))
                    performance_metrics[key] = init
                performance_metrics[key][i, j] = new_performance_metrics[key]

    # plot: perf vs t
    for key in performance_metrics:
        matrix_perfs = performance_metrics[key]
        for i, f_str in enumerate(features_str):
            plt.semilogx(t_list, matrix_perfs[i, :], label=f_str)
        plt.legend(bbox_to_anchor=(0, 1.02), loc='lower left')
        plt.xlabel('t')
        plt.ylabel(key)
        plt.draw()  # re-render
        path = '_vs_'
        if mean_transfo:
            path += 'mean_'
        if rotation_transfo:
            path += 'rotation_'
        path += 'transfo_'
        path = os.path.join(folder, key + path)
        plt.savefig(path, bbox_inches='tight')
        tikzplotlib.save(path + '.tex')
        plt.close('all')


if __name__ == '__main__':
    SEED = 0
    N_JOBS = -1
    DATASET = 'train_small_test'
    N_POINTS = 10
    t_LIST = np.geomspace(5*1e-4, 5*1e-2, num=N_POINTS, endpoint=True)
    MIN_GRAD_NORM_MEAN = 1e-4

    print('seed:', SEED)
    print('n_jobs:', N_JOBS)
    print()

    FEATURES = [
        identity_euclidean(),
        mean_vector_euclidean(),
        covariance_euclidean(assume_centered=True),
        covariance_div_alpha(
            alpha=0, div_alpha_real_case=True, symmetrize_div=True,
            min_grad_norm_mean=MIN_GRAD_NORM_MEAN
        ),
        location_covariance_div_alpha(
            alpha=0, div_alpha_real_case=True, symmetrize_div=True,
            min_grad_norm_mean=MIN_GRAD_NORM_MEAN
        ),
        location_covariance_texture_constrained_texture_div_alpha(
            iter_max=300,
            information_geometry=True,
            solver='steepest',
            reg_type='L2',
            reg_beta=1e-11,
            reg_kappa='trace_SCM',
            alpha=0,
            div_alpha_real_case=True,
            symmetrize_div=True,
            min_grad_norm_mean=MIN_GRAD_NORM_MEAN
        )
    ]

    # simu 1: loc+cov+text with mean transformation
    rnd.seed(SEED)

    bzh_class = Bzh_loader(
        dataset=DATASET,
        load_npy=True,
        level='L1C',
        verbose=True
    )

    main(
        dataset=bzh_class,
        t_list=t_LIST,
        mean_transfo=True,
        rotation_transfo=False,
        features=FEATURES,
        n_jobs=N_JOBS,
        verbose=True
    )

    # simu 2: loc+cov+text with rotation transformation
    rnd.seed(SEED)

    bzh_class = Bzh_loader(
        dataset=DATASET,
        load_npy=True,
        level='L1C',
        verbose=True
    )

    main(
        dataset=bzh_class,
        t_list=t_LIST,
        mean_transfo=False,
        rotation_transfo=True,
        features=FEATURES,
        n_jobs=N_JOBS,
        verbose=True
    )

    # simu 3: loc+cov+text with mean+rotation transformation
    rnd.seed(SEED)

    bzh_class = Bzh_loader(
        dataset=DATASET,
        load_npy=True,
        level='L1C',
        verbose=True
    )

    main(
        dataset=bzh_class,
        t_list=t_LIST,
        mean_transfo=True,
        rotation_transfo=True,
        features=FEATURES,
        n_jobs=N_JOBS,
        verbose=True
    )
