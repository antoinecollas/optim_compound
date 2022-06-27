import autograd.numpy as np
import autograd.numpy.random as rnd
import os
import sys

from classification.time_series import Bzh_loader


FULL_TEST = os.getenv('FULL_TEST', 'false').lower() == 'true'

if FULL_TEST:
    def test_data_loader_bzh():
        seed = 0
        rnd.seed(seed)

        dataset_size = 608263
        dataset_train_size = 485649
        dataset_test_size = dataset_size - dataset_train_size
        K = 9
        N = 45
        p = 13

        bzh_class = Bzh_loader(
            dataset='train_test',
            path_npy='tests_breizhcrops_npy',
            load_npy=False,
            level='L1C',
            verbose=True
        )

        # get_info
        assert bzh_class.get_info() == (p, N, K)

        # print_info
        sys.stdout = open(os.devnull, 'w')
        bzh_class.print_info()
        sys.stdout = sys.__stdout__

        # train
        X_train, y_train = bzh_class.get_data_train()
        assert type(X_train) == np.ndarray
        assert X_train.shape == (dataset_train_size, p, N)
        assert type(y_train) == np.ndarray
        assert y_train.shape == (dataset_train_size,)
        assert len(np.unique(y_train)) == K

        # test
        X_test, y_test = bzh_class.get_data_test()
        assert type(X_test) == np.ndarray
        assert X_test.shape == (dataset_test_size, p, N)
        assert type(y_test) == np.ndarray
        assert y_test.shape == (dataset_test_size,)
        assert len(np.unique(y_test)) == K

        # save data into npy
        bzh_class.save_into_npy()

        # load dataset using saved npy
        bzh_class = Bzh_loader(
            dataset='train_test',
            path_npy='tests_breizhcrops_npy',
            load_npy=True,
            level='L1C',
            verbose=True
        )

        # get_info
        assert bzh_class.get_info() == (p, N, K)

        # print_info
        sys.stdout = open(os.devnull, 'w')
        bzh_class.print_info()
        sys.stdout = sys.__stdout__

        # train
        X_train, y_train = bzh_class.get_data_train()
        assert type(X_train) == np.ndarray
        assert X_train.shape == (dataset_train_size, p, N)
        assert type(y_train) == np.ndarray
        assert y_train.shape == (dataset_train_size,)
        assert len(np.unique(y_train)) == K

        # test
        X_test, y_test = bzh_class.get_data_test()
        assert type(X_test) == np.ndarray
        assert X_test.shape == (dataset_test_size, p, N)
        assert type(y_test) == np.ndarray
        assert y_test.shape == (dataset_test_size,)
        assert len(np.unique(y_test)) == K
