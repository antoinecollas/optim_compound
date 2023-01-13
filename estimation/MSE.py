import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import os
from pyCovariance import monte_carlo
from pyCovariance.evaluation import create_directory
from pyCovariance.features.base import Feature, make_feature_prototype
from pyCovariance.features.covariance_texture import\
        Tyler_estimator_normalized_det
from pyCovariance.features.location_covariance_texture import\
        estimate_compound_Gaussian_constrained_texture,\
        Gaussian_estimation_constrained_texture,\
        Tyler_estimator_unknown_location_constrained_scatter
from pyCovariance.generation_data import\
        generate_covariance,\
        generate_textures_gamma_dist,\
        sample_compound_distribution
from pymanopt.manifolds import ComplexEuclidean, HermitianPositiveDefinite
import tikzplotlib
from tqdm import tqdm


def normalize_scatter_matrix(scatter):
    return scatter / (la.det(scatter)**(1/scatter.shape[-1]))


# define a wrapper that normalizes
# the covariance matrix once estimated
def wrapper_normalize_scatter_matrix(estimator):
    def _new_estimator(*args, **kwargs):
        point = list(estimator(*args, **kwargs))
        point[1] = normalize_scatter_matrix(point[1])
        return tuple(point)
    return _new_estimator


def get_M(scatter_use_SPD_dist, p, N):
    if scatter_use_SPD_dist:
        M = (ComplexEuclidean, HermitianPositiveDefinite, ComplexEuclidean)
        args_M = {
            'sizes': ((p, 1), p, (N, 1)),
            'weights': (1, 1, 1)
        }
    else:
        M = (ComplexEuclidean, ComplexEuclidean, ComplexEuclidean)
        args_M = {
            'sizes': ((p, 1), (p, p), (N, 1)),
            'weights': (1, 1, 1)
        }
    return M, args_M


@make_feature_prototype
def Gaussian(scatter_use_SPD_dist, p=None, N=None):
    name = 'Gaussian'
    M, args_M = get_M(scatter_use_SPD_dist, p, N)
    return Feature(
        name,
        wrapper_normalize_scatter_matrix(
            Gaussian_estimation_constrained_texture),
        M,
        args_M
    )


@make_feature_prototype
def Tyler_known_location(mu, iter_max, scatter_use_SPD_dist, p=None, N=None):
    if mu is None:
        mu_known = False
        name = 'Tyler_Gaussian_location'
    else:
        mu_known = True
        name = 'Tyler_known_location'

    M, args_M = get_M(scatter_use_SPD_dist, p, N)

    def _Tyler(X):
        if mu_known:
            location = mu
        else:
            location = np.mean(X, axis=1, keepdims=True)
        X = X - location
        sigma, tau, _, _ = Tyler_estimator_normalized_det(
            X, iter_max=iter_max)
        c = np.exp(np.mean(np.log(tau)))
        sigma = sigma * c
        tau = tau / c
        return location, sigma, tau

    return Feature(
        name,
        wrapper_normalize_scatter_matrix(_Tyler),
        M,
        args_M
    )


@make_feature_prototype
def Tyler_unknown_location(iter_max, scatter_use_SPD_dist, p=None, N=None):
    name = 'Tyler_unknown_location'

    M, args_M = get_M(scatter_use_SPD_dist, p, N)

    def _Tyler(X):
        res = Tyler_estimator_unknown_location_constrained_scatter(
            X, iter_max=iter_max)
        mu, sigma, tau, _, _ = res
        c = np.exp(np.mean(np.log(tau)))
        sigma = sigma * c
        tau = tau / c
        return mu, sigma, tau

    return Feature(
        name,
        wrapper_normalize_scatter_matrix(_Tyler),
        M,
        args_M
    )


@make_feature_prototype
def Riemannian_opt(
    solver='conjugate',
    information_geometry=False,
    min_grad_norm=0,
    min_step_size=-np.inf,
    iter_max=np.inf,
    time_max=np.inf,
    scatter_use_SPD_dist=True,
    p=None,
    N=None
):
    name = solver
    if information_geometry:
        name += '_IG'

    M, args_M = get_M(scatter_use_SPD_dist, p, N)

    def _estimation(X):
        mu, sigma, tau, _ = estimate_compound_Gaussian_constrained_texture(
            X,
            init=None,
            information_geometry=information_geometry,
            reg_type='L2',
            reg_beta=0,
            min_grad_norm=min_grad_norm,
            min_step_size=min_step_size,
            iter_max=iter_max,
            time_max=time_max,
            autodiff=True,
            solver=solver
        )
        return mu, sigma, tau

    return Feature(
        name,
        wrapper_normalize_scatter_matrix(_estimation),
        M,
        args_M
    )


def main(
    n_MC,
    p,
    nu,
    list_n_samples,
    iter_max,
    min_grad_norm,
    min_step_size,
    solvers_IG,
    scatter_use_SPD_dist,
    n_jobs,
    verbose=True
):
    matplotlib.use('Agg')

    if verbose:
        print('\n###########################################################')
        print('############## MSE versus the number of data ##############')
        print('###########################################################\n')

    seed = 0
    rnd.seed(seed)
    if verbose:
        print('seed:', seed)

    # path to save plot
    folder = os.path.join('estimation')
    folder = create_directory(folder)

    # generate location and scatter matrix
    mu = rnd.normal(loc=0, scale=1, size=(p, 1))
    sigma = generate_covariance(p)

    # plot eigenvalues of sigma
    eigv = la.eigvalsh(sigma)[::-1]
    plt.semilogy(eigv)
    path_temp = os.path.join(folder, 'eigenvalues')
    plt.savefig(path_temp)
    plt.close('all')

    # sample fct
    def sample_fct(n):
        tau = generate_textures_gamma_dist(
            n, nu=nu, min_value=1e-16, unit_prod=True)
        return sample_compound_distribution(tau, sigma) + mu

    features_list = [
        Gaussian(scatter_use_SPD_dist),
        # Tyler_known_location(mu=None, iter_max=iter_max,
        #                      scatter_use_SPD_dist=scatter_use_SPD_dist),
        Tyler_known_location(mu=mu, iter_max=iter_max,
                             scatter_use_SPD_dist=scatter_use_SPD_dist),
        Tyler_unknown_location(iter_max=iter_max,
                               scatter_use_SPD_dist=scatter_use_SPD_dist)
    ]
    features_list += [
        Riemannian_opt(
            solver=s,
            information_geometry=True,
            iter_max=iter_max,
            min_grad_norm=min_grad_norm,
            min_step_size=min_step_size,
            scatter_use_SPD_dist=scatter_use_SPD_dist
        ) for s in solvers_IG
    ]

    n_distances = 4
    shape = (len(features_list), n_distances, len(list_n_samples))
    mean_errors = np.zeros(shape)

    list_n_samples_iterator = list_n_samples
    if verbose:
        list_n_samples_iterator = tqdm(list_n_samples_iterator)

    for i, n in enumerate(list_n_samples_iterator):
        true_parameters = [
            mu,
            normalize_scatter_matrix(sigma),
            np.ones((n, 1))
        ]
        mean_errors[:, :, i] = monte_carlo(
            true_parameters,
            partial(sample_fct, n=n),
            features_list,
            n_MC,
            n_jobs=n_jobs,
            verbose=False
        )

    # plot MSE of location estimation
    plt.figure()
    for i, f in enumerate(features_list):
        plt.loglog(list_n_samples, mean_errors[i][1],
                   label=str(f(1, 1)), marker='+')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('MSE - location')
    plt.grid(visible=True, which='both')
    path_temp = os.path.join(folder, 'N_MSE_mu')
    plt.savefig(path_temp)
    tikzplotlib.save(path_temp + '.tex')
    plt.close('all')

    # plot MSE of scatter matrix estimation
    plt.figure()
    for i, f in enumerate(features_list):
        plt.loglog(list_n_samples, mean_errors[i][2],
                   label=str(f(1, 1)), marker='+')
    plt.legend()
    plt.xlabel('N')
    if scatter_use_SPD_dist:
        plt.ylabel('MSE (Riemannian) - scatter matrix')
    else:
        plt.ylabel('MSE (Euclidean) - scatter matrix')
    plt.grid(visible=True, which='both')
    path_temp = os.path.join(folder, 'N_MSE_scatter')
    plt.savefig(path_temp)
    tikzplotlib.save(path_temp + '.tex')
    plt.close('all')


if __name__ == '__main__':
    n_MC = 2000
    p = 10
    nu = 0.1
    list_n_samples = np.geomspace(2*p, 100*p, num=10, dtype=int)
    iter_max = 500
    min_grad_norm = 1e-5
    min_step_size = 1e-10
    solvers_IG = ['steepest']
    n_jobs = -1
    verbose = True

    main(
        n_MC=n_MC,
        p=p,
        nu=nu,
        list_n_samples=list_n_samples,
        iter_max=iter_max,
        min_grad_norm=min_grad_norm,
        min_step_size=min_step_size,
        solvers_IG=solvers_IG,
        scatter_use_SPD_dist=False,
        n_jobs=n_jobs,
        verbose=verbose
    )

    main(
        n_MC=n_MC,
        p=p,
        nu=nu,
        list_n_samples=list_n_samples,
        iter_max=iter_max,
        min_grad_norm=min_grad_norm,
        min_step_size=min_step_size,
        solvers_IG=solvers_IG,
        scatter_use_SPD_dist=True,
        n_jobs=n_jobs,
        verbose=verbose
    )
