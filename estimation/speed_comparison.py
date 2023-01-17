import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
import matplotlib
import matplotlib.pyplot as plt
import os
from pyCovariance.evaluation import create_directory
from pyCovariance.features.location_covariance_texture import\
        estimate_compound_Gaussian_constrained_texture
from pyCovariance.generation_data import\
        generate_covariance,\
        generate_textures_gamma_dist,\
        sample_compound_distribution
from tqdm import tqdm


def main(
    N,
    p,
    nu,
    reg_types,
    reg_betas,
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
    folder = os.path.join('estimation')
    folder = create_directory(folder)

    # generate location, scatter matrix, textures
    mu = rnd.normal(loc=0, scale=1, size=(p, 1))
    sigma = generate_covariance(p)

    # generate and enforce textures to be above a threshold
    tau = generate_textures_gamma_dist(
        N, nu=nu, unit_prod=True, min_value=1e-16)

    # plot histograms of textures
    plt.hist(tau, bins=100)
    path_temp = os.path.join(folder, 'hist_textures')
    plt.savefig(path_temp)
    plt.close('all')

    plt.hist(np.log10(tau), bins=100)
    path_temp = os.path.join(folder, 'hist_textures_log10')
    plt.savefig(path_temp)
    plt.close('all')

    # plot eigenvalues of sigma
    eigv = la.eigvalsh(sigma)[::-1]
    plt.semilogy(eigv)
    path_temp = os.path.join(folder, 'eigenvalues')
    plt.savefig(path_temp)
    plt.close('all')

    # sample fct
    def sample_fct():
        return sample_compound_distribution(tau, sigma) + mu

    def estimation(X, reg_type, reg_beta, s, IG):
        _, _, _, log = estimate_compound_Gaussian_constrained_texture(
            X,
            information_geometry=IG,
            reg_type=reg_type,
            reg_beta=reg_beta,
            min_step_size=-np.inf,
            iter_max=iter_max,
            time_max=np.inf,
            solver=s,
            autodiff=True
        )
        return log

    X = sample_fct()

    if verbose:
        iterator = tqdm(reg_types)
    else:
        iterator = reg_types

    for reg_type in iterator:
        # prepare plot
        # two lines for cost fct and grad norm
        # one column per reg_beta
        fig = plt.figure(figsize=(8, 3.6))
        gs = fig.add_gridspec(2, len(reg_betas), hspace=0.15, wspace=0.1)
        axes_cst_fct, axes_grad_norm = gs.subplots(sharex='col', sharey='row')
        YLABEL_COORDS = (-0.3, 0.5)
        XTICKS = [1, 10, 100, 1000]

        for k, (ax_cst_fct, ax_grad_norm, reg_beta) in enumerate(
                zip(axes_cst_fct, axes_grad_norm, reg_betas)):
            # log scale
            ax_cst_fct.set_xscale('log')
            ax_cst_fct.set_yscale('log')
            ax_grad_norm.set_xscale('log')
            ax_grad_norm.set_yscale('log')

            # estimation
            res = list()
            for s in solvers:
                res.append(estimation(X, reg_type, reg_beta, s, IG=False))
            res_IG = list()
            for s in solvers_IG:
                res_IG.append(estimation(X, reg_type, reg_beta, s, IG=True))

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
                ax_cst_fct.set_ylabel('Regularized NLL')
            ax_cst_fct.set_xticks(XTICKS)
            # remove scientific notation from xaxis of ax_cst_fct
            ticker = matplotlib.ticker.StrMethodFormatter('{x}')
            ax_cst_fct.xaxis.set_minor_formatter(ticker)
            ax_cst_fct.xaxis.set_major_formatter(ticker)
            # remove scientific notation from yaxis of ax_cst_fct
            ticker = matplotlib.ticker.StrMethodFormatter('{x:.0f}')
            ax_cst_fct.yaxis.set_minor_formatter(ticker)
            ax_cst_fct.yaxis.set_major_formatter(ticker)
            if reg_beta == 0:
                ax_cst_fct.set_title(r'$\beta = 0$')
            else:
                ax_cst_fct.set_title(
                    r'$\beta = 10^{' + str(int(np.log10(reg_beta))) + '}$')
            ax_cst_fct.grid(visible=True, which='both')

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
            ax_grad_norm.set_xlabel('Iterations')
            ax_grad_norm.grid(visible=True, which='major')

        filename = 'likelihood_gradnorm_vs_iterations_'+reg_type+'.pdf'
        path_temp = os.path.join(folder, filename)
        plt.savefig(path_temp, bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    main(
        N=150,
        p=10,
        nu=1,
        reg_types=['L2'],
        reg_betas=[0, 1e-5, 1e-3],
        iter_max=1000,
        solvers=['steepest', 'conjugate'],
        solvers_IG=['steepest'],
    )
