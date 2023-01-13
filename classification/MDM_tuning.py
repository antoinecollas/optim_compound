import autograd.numpy as np
import autograd.numpy.random as rnd
import itertools
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import os
import tikzplotlib

from pyCovariance.classification import MDM
from pyCovariance.evaluation import create_directory
from pyCovariance.features.location_covariance_texture import\
        location_covariance_texture_constrained_texture_div_alpha

from classification.time_series import Bzh_loader, compute_performance_metrics


def main(
    dataset,
    alpha_list,
    symmetrize_div,
    reg_dict,
    min_grad_norm_mean,
    n_jobs,
    verbose=True
):
    matplotlib.use('Agg')

    features = [
        partial(
            location_covariance_texture_constrained_texture_div_alpha,
            div_alpha_real_case=True,
            symmetrize_div=symmetrize_div,
            min_grad_norm_mean=min_grad_norm_mean
        )
    ]

    # folder path to save files
    folder_name = 'MDM_tuning_' + dataset.dataset + '_' + dataset.level
    folder_name = os.path.join('classification', folder_name)
    folder = create_directory(folder_name)

    X_train, y_train = dataset.get_data_train()
    X_test, y_test = dataset.get_data_test()

    if verbose:
        dataset.print_info()

    reg_types = list(reg_dict.keys())

    # MDM
    for feature in features:

        for alpha in alpha_list:
            # store all results
            performance_metrics = dict()
            reg_betas = list()

            for reg_type in reg_types:
                reg_betas.append(reg_dict[reg_type])
                temp_performance_metrics = dict()

                for reg_beta in reg_betas[len(reg_betas)-1]:
                    if verbose:
                        print('Regularization:', reg_type, reg_beta)
                        print('Alpha divergence:', alpha)

                    # MDM
                    clf = MDM(
                        feature=feature(
                            reg_type=reg_type,
                            reg_beta=reg_beta,
                            reg_kappa='trace_SCM',
                            alpha=alpha,
                        ),
                        n_jobs=n_jobs,
                        verbose=verbose
                    )
                    y_pred = clf.fit_predict(X_train, y_train)
                    tmp = compute_performance_metrics(
                        y_train, y_pred,
                        classnames=dataset.classname, verbose=verbose)
                    if verbose:
                        print('Overall accuracy on train set:', tmp['OA'])
                        print('Average accuracy on train set:', tmp['AA'])

                    # evaluation
                    y_pred = clf.predict(X_test)
                    new_performance_metrics = compute_performance_metrics(
                        y_test, y_pred,
                        classnames=dataset.classname, verbose=verbose)
                    for key in new_performance_metrics:
                        if not (key in temp_performance_metrics):
                            temp_performance_metrics[key] = list()
                        to_append = new_performance_metrics[key]
                        temp_performance_metrics[key].append(to_append)
                    if verbose:
                        OA = new_performance_metrics['OA']
                        print('Overall accuracy on test set:', OA)
                        AA = new_performance_metrics['AA']
                        print('Average accuracy on test set:', AA)
                        print()

                for key in temp_performance_metrics:
                    if not (key in performance_metrics):
                        performance_metrics[key] = list()
                    to_append = temp_performance_metrics[key]
                    performance_metrics[key].append(to_append)

            alpha = str(alpha).replace('.', '_')
            markers_symbols = ('d', 'p', '*', 'v', 'X', 'v')

            # plot OAs
            for key in performance_metrics:
                metric = performance_metrics[key]
                markers = itertools.cycle(markers_symbols)
                for i in range(len(reg_types)):
                    plt.semilogx(reg_betas[i], np.array(metric[i])*100,
                                 marker=next(markers), label=reg_types[i])
                plt.legend(fontsize=14)
                tmp = '_alpha_'
                if symmetrize_div:
                    tmp += 'sym_'
                path = os.path.join(folder, key + tmp + alpha)
                plt.savefig(path)
                tikzplotlib.save(path + '.tex')
                plt.close('all')

            # table of OAs and AAs
            table_OAs = np.zeros(len(reg_types))
            table_AAs = np.zeros(len(reg_types))
            table_betas = np.zeros(len(reg_types))
            reg_str = list()
            OAs = performance_metrics['OA']
            AAs = performance_metrics['AA']
            for i, reg_type in enumerate(reg_types):
                k = np.argmax(OAs[i])
                table_OAs[i] = OAs[i][k]
                table_AAs[i] = AAs[i][k]
                table_betas[i] = reg_betas[i][k]
                reg_str.append(reg_type+'_'+str(table_betas[i]))
            results = np.stack([table_OAs, table_AAs], axis=-1)
            column_labels = ['OA', 'AA']
            plt.table(cellText=results, rowLabels=reg_str,
                      colLabels=column_labels, loc='center')
            plt.axis('off')
            tmp = 'OA_AA_table_alpha_'
            if symmetrize_div:
                tmp += 'sym_'
            path = os.path.join(folder, tmp + alpha)
            plt.savefig(path, bbox_inches='tight')
            plt.close('all')


if __name__ == '__main__':
    SEED = 0
    N_JOBS = -1
    DATASET = 'train_val_small'
    ALPHA_LIST = [0]
    SYMMETRIZE_DIV = True
    N_POINTS = 10
    MIN_GRAD_NORM_MEAN = 1e-4

    print('seed:', SEED)
    print('n_jobs:', N_JOBS)
    print('dataset:', DATASET)
    print()

    REG_DICT = {
        # 'L1': np.geomspace(5*1e-8, 5*1e-4, num=N_POINTS),
        'L2': np.geomspace(1e-20, 1e-6, num=N_POINTS),
        # 'BW': np.geomspace(5*1e-8, 5*1e-4, num=N_POINTS),
        # 'KL': np.geomspace(1e-6, 1e-3, num=N_POINTS),
    }

    rnd.seed(SEED)

    bzh_class = Bzh_loader(
        dataset=DATASET,
        load_npy=True,
        level='L1C',
        verbose=True
    )

    main(
        dataset=bzh_class,
        alpha_list=ALPHA_LIST,
        symmetrize_div=SYMMETRIZE_DIV,
        reg_dict=REG_DICT,
        min_grad_norm_mean=MIN_GRAD_NORM_MEAN,
        n_jobs=N_JOBS
    )
