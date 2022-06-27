import autograd.numpy as np
import autograd.numpy.random as rnd
import matplotlib
import matplotlib.pyplot as plt
import os
from pyCovariance.evaluation import create_directory
from pyCovariance.features.base import _FeatureArray
from pyCovariance.generation_data import\
        generate_covariance,\
        generate_textures_gamma_dist
import tikzplotlib
from tqdm import tqdm

from center_of_mass.computation import\
        estimate_center_of_mass_constrained_texture


def main(
    Ms,
    N,
    p,
    nu,
    iter_max,
    solvers,
    solvers_IG,
    verbose=True
):
    matplotlib.use('Agg')

    if verbose:
        print('\n###########################################################')
        print('### Cost function according to the number of iterations ###')
        print('###########################################################\n')

    seed = 0
    rnd.seed(seed)
    if verbose:
        print('seed:', seed)

    # path to save plot
    folder = os.path.join('center_of_mass')
    folder = create_directory(folder)

    if verbose:
        iterator = tqdm(Ms)
    else:
        iterator = Ms

    for M in iterator:
        # generate location, scatter matrix, textures
        theta = _FeatureArray((p, 1), (p, p), (N, 1))
        for i in range(M):
            mu = rnd.normal(loc=0, scale=1, size=(p, 1))
            sigma = generate_covariance(p)
            tau = generate_textures_gamma_dist(
                N, nu=nu, unit_prod=True, min_value=1e-16)
            theta.append([mu, sigma, tau])

        def estimation(X, solver, IG):
            _, _, _, log = estimate_center_of_mass_constrained_texture(
                X,
                information_geometry=IG,
                min_step_size=-np.inf,
                iter_max=iter_max,
                time_max=np.inf,
                solver=s
            )
            return log

        # estimation
        res = list()
        for s in solvers:
            res.append(estimation(theta, solver=s, IG=False))
        res_IG = list()
        for s in solvers_IG:
            res_IG.append(estimation(theta, solver=s, IG=True))

        # plot cost fct vs iterations
        min_value = np.inf
        for i, s in enumerate(solvers):
            tmp = np.min(res[i]['iterations']['f(x)'])
            min_value = np.min([tmp, min_value])
        for i, s in enumerate(solvers_IG):
            tmp = np.min(res_IG[i]['iterations']['f(x)'])
            min_value = np.min([tmp, min_value])
        for i, s in enumerate(solvers):
            to_plot = res[i]['iterations']['f(x)'] - min_value + 1
            plt.loglog(np.arange(1, len(to_plot) + 1), to_plot,
                       label=s, marker='')
        for i, s in enumerate(solvers_IG):
            to_plot = res_IG[i]['iterations']['f(x)'] - min_value + 1
            plt.loglog(np.arange(1, len(to_plot) + 1), to_plot,
                       label=s+' IG', marker='')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Center of mass - cost function')
        plt.grid(visible=False, which='both')
        filename = 'cost_fct_M_' + str(M)
        path_temp = os.path.join(folder, filename)
        plt.savefig(path_temp)
        tikzplotlib.save(path_temp + '.tex')
        plt.close('all')

        # plot grad norm vs iterations
        for i, s in enumerate(solvers):
            to_plot = res[i]['iterations']['gradnorm']
            plt.loglog(np.arange(1, len(to_plot) + 1), to_plot,
                       label=s, marker='')
        for i, s in enumerate(solvers_IG):
            to_plot = res_IG[i]['iterations']['gradnorm']
            plt.loglog(np.arange(1, len(to_plot) + 1), to_plot,
                       label=s+' IG', marker='')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Gradient norm')
        plt.grid(visible=False, which='both')
        filename = 'gradnorm_M_' + str(M)
        path_temp = os.path.join(folder, filename)
        plt.savefig(path_temp)
        tikzplotlib.save(path_temp + '.tex')
        plt.close('all')


if __name__ == '__main__':
    Ms = [2, 10, 100]
    N = 150
    p = 10
    nu = 1
    iter_max = 1000
    solvers = ['steepest', 'conjugate']
    solvers_IG = ['steepest']

    main(
        Ms=Ms,
        N=N,
        p=p,
        nu=nu,
        iter_max=iter_max,
        solvers=solvers,
        solvers_IG=solvers_IG,
    )
