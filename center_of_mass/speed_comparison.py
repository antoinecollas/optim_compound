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

    # prepare plot
    # two lines for cost fct and grad norm
    # one column per M
    fig = plt.figure(figsize=(8, 3.6))
    gs = fig.add_gridspec(2, len(Ms), hspace=0.15, wspace=0.1)
    axes_cst_fct, axes_grad_norm = gs.subplots(sharex='col', sharey='row')
    YLABEL_COORDS = (-0.3, 0.5)
    XTICKS = [1, 10, 100, 1000]

    if verbose:
        iterator = enumerate(zip(tqdm(axes_cst_fct), axes_grad_norm, Ms))
    else:
        iterator = enumerate(zip(axes_cst_fct, axes_grad_norm, Ms))

    for k, (ax_cst_fct, ax_grad_norm, M) in iterator:
        # log scale
        ax_cst_fct.set_xscale('log')
        ax_cst_fct.set_yscale('log')
        ax_grad_norm.set_xscale('log')
        ax_grad_norm.set_yscale('log')

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
            to_plot_y = res[i]['iterations']['f(x)'] - min_value + 1
            to_plot_x = np.arange(1, len(to_plot_y) + 1)
            ax_cst_fct.plot(
                to_plot_x, to_plot_y, label='plain '+s, marker='')
        for i, s in enumerate(solvers_IG):
            to_plot_y = res_IG[i]['iterations']['f(x)'] - min_value + 1
            to_plot_x = np.arange(1, len(to_plot_y) + 1)
            ax_cst_fct.plot(
                to_plot_x, to_plot_y, label='proposed algo.', marker='')
        if k == 0:
            ax_cst_fct.legend(fontsize=6, loc='upper right')
            ax_cst_fct.yaxis.set_label_coords(*YLABEL_COORDS)
            ax_cst_fct.set_ylabel('Cost function')
        ax_cst_fct.set_xticks(XTICKS)
        # remove scientific notation from xaxis of ax_cst_fct
        ticker = matplotlib.ticker.StrMethodFormatter('{x}')
        ax_cst_fct.xaxis.set_minor_formatter(ticker)
        ax_cst_fct.xaxis.set_major_formatter(ticker)
        ax_cst_fct.set_yticks(1e1**np.arange(0, 9, 2))
        ax_cst_fct.set_title(r'$M = '+str(M)+'$')
        ax_cst_fct.grid(visible=True, which='major')

        # plot grad norm vs iterations
        for i, s in enumerate(solvers):
            to_plot_y = res[i]['iterations']['gradnorm']
            to_plot_x = np.arange(1, len(to_plot_y) + 1)
            ax_grad_norm.plot(
                to_plot_x, to_plot_y, label='plain '+s, marker='')
        for i, s in enumerate(solvers_IG):
            to_plot_y = res_IG[i]['iterations']['gradnorm']
            to_plot_x = np.arange(1, len(to_plot_y) + 1)
            ax_grad_norm.plot(
                to_plot_x, to_plot_y, label='proposed algo.', marker='')
        if k == 0:
            ax_grad_norm.yaxis.set_label_coords(*YLABEL_COORDS)
            ax_grad_norm.set_ylabel('Gradient norm')
        ax_grad_norm.set_xticks(XTICKS)
        ax_grad_norm.set_yticks(1e1**np.arange(-5, 8, 2))
        ax_grad_norm.set_xlabel('Iterations')
        ax_grad_norm.grid(visible=True, which='major')

    filename = 'center_of_mass_cost_gradnorm_vs_iterations.pdf'
    path_temp = os.path.join(folder, filename)
    plt.savefig(path_temp, bbox_inches='tight')
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
