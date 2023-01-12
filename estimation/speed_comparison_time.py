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
import tikzplotlib
from tqdm import tqdm


def main(
    N,
    p,
    nu,
    reg_types,
    reg_betas,
    time_max,
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
            iter_max=np.inf,
            time_max=time_max,
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
        folder_reg = os.path.join(folder, reg_type)
        if not os.path.exists(folder_reg):
            os.makedirs(folder_reg, exist_ok=True)

        for reg_beta in reg_betas:
            reg_beta_str = str(reg_beta).replace('.', '_')

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
                to_plot_x = res[i]['iterations']['time']
                to_plot_x -= np.min(to_plot_x)
                to_plot_y = res[i]['iterations']['f(x)'] - min_value + 1
                plt.plot(to_plot_x, to_plot_y, label=s, marker='')
                plt.xscale('symlog', linthresh=to_plot_x[1])
                plt.yscale('log')
            for i, s in enumerate(solvers_IG):
                to_plot_x = res_IG[i]['iterations']['time']
                to_plot_x -= np.min(to_plot_x)
                to_plot_y = res_IG[i]['iterations']['f(x)'] - min_value + 1
                plt.plot(to_plot_x, to_plot_y, label=s+' IG', marker='')
                plt.xscale('symlog', linthresh=to_plot_x[1])
                plt.yscale('log')
            plt.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Negative log-likelihood')
            plt.grid(visible=False, which='both')
            filename = 'time_likelihood_beta_' + reg_beta_str
            path_temp = os.path.join(folder_reg, filename)
            plt.savefig(path_temp)
            tikzplotlib.save(path_temp + '.tex')
            plt.close('all')

            # plot grad norm vs iterations
            for i, s in enumerate(solvers):
                to_plot_x = res[i]['iterations']['time']
                to_plot_x -= np.min(to_plot_x)
                to_plot_y = res[i]['iterations']['gradnorm']
                plt.plot(to_plot_x, to_plot_y, label=s, marker='')
                plt.xscale('symlog', linthresh=to_plot_x[1])
                plt.yscale('log')
            for i, s in enumerate(solvers_IG):
                to_plot_x = res_IG[i]['iterations']['time']
                to_plot_x -= np.min(to_plot_x)
                to_plot_y = res_IG[i]['iterations']['gradnorm']
                plt.plot(to_plot_x, to_plot_y, label=s+' IG', marker='')
                plt.xscale('symlog', linthresh=to_plot_x[1])
                plt.yscale('log')
            plt.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Gradient norm')
            plt.grid(visible=False, which='both')
            filename = 'time_gradnorm_likelihood_beta_' + reg_beta_str
            path_temp = os.path.join(folder_reg, filename)
            plt.savefig(path_temp)
            tikzplotlib.save(path_temp + '.tex')
            plt.close('all')


if __name__ == '__main__':
    main(
        N=150,
        p=10,
        nu=1,
        reg_types=['L2'],
        reg_betas=[0, 1e-5, 1e-3],
        time_max=1,
        solvers=['steepest', 'conjugate'],
        solvers_IG=['steepest'],
    )
