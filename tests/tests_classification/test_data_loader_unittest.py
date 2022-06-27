import autograd.numpy as np
import autograd.numpy.random as rnd
from numpy import testing as np_testing
import os
import sys

from classification.time_series import Bzh_loader


def test_data_loader_unittest():
    seed = 0
    rnd.seed(seed)

    K = 7  # nb of classes in "Belle-ile" dataset
    dataset_train_size = 1049
    dataset_test_size = 1049
    N = 45
    p = 13

    # test get_list_available_datasets
    tmp = Bzh_loader.get_list_available_datasets()
    assert type(tmp) == list
    for t in tmp:
        assert type(t) == str

    bzh_class = Bzh_loader(dataset='unittest',
                           path_npy='tests_breizhcrops_npy',
                           load_npy=False, verbose=False)

    # get_info
    tmp = bzh_class.get_info()
    assert tmp[0] == p
    assert tmp[1] == N
    assert tmp[2] == K

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
    bzh_class = Bzh_loader(dataset='unittest',
                           path_npy='tests_breizhcrops_npy',
                           load_npy=True, verbose=False)

    # get_info
    tmp = bzh_class.get_info()
    assert tmp[0] == p
    assert tmp[1] == N
    assert tmp[2] == K

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


def test_data_loader_unittest_small():
    seed = 0
    rnd.seed(seed)

    K = 7  # nb of classes in "Belle-ile" dataset
    N = 45
    p = 13

    bzh_class = Bzh_loader(dataset='unittest_small',
                           path_npy='tests_breizhcrops_npy',
                           load_npy=False, verbose=False)

    # get_info
    tmp = bzh_class.get_info()
    assert tmp[0] == p
    assert tmp[1] == N
    assert tmp[2] == K

    # print_info
    sys.stdout = open(os.devnull, 'w')
    bzh_class.print_info()
    sys.stdout = sys.__stdout__

    # train
    X_train, y_train = bzh_class.get_data_train()
    assert type(X_train) == np.ndarray
    assert X_train.shape[1:] == (p, N)
    assert type(y_train) == np.ndarray
    assert len(y_train) == len(X_train)
    assert len(np.unique(y_train)) == K

    # test
    X_test, y_test = bzh_class.get_data_test()
    assert type(X_test) == np.ndarray
    assert X_test.shape[1:] == (p, N)
    assert type(y_test) == np.ndarray
    assert len(y_test) == len(X_test)
    assert len(np.unique(y_test)) == K

    # save data into npy
    bzh_class.save_into_npy()

    # load dataset using saved npy
    bzh_class = Bzh_loader(dataset='unittest_small',
                           path_npy='tests_breizhcrops_npy',
                           load_npy=True, verbose=False)

    # get_info
    tmp = bzh_class.get_info()
    assert tmp[0] == p
    assert tmp[1] == N
    assert tmp[2] == K

    # print_info
    sys.stdout = open(os.devnull, 'w')
    bzh_class.print_info()
    sys.stdout = sys.__stdout__

    # train
    X_train, y_train = bzh_class.get_data_train()
    assert type(X_train) == np.ndarray
    assert X_train.shape[1:] == (p, N)
    assert type(y_train) == np.ndarray
    assert len(y_train) == len(X_train)
    assert len(np.unique(y_train)) == K

    # test
    X_test, y_test = bzh_class.get_data_test()
    assert type(X_test) == np.ndarray
    assert X_test.shape[1:] == (p, N)
    assert type(y_test) == np.ndarray
    assert len(y_test) == len(X_test)
    assert len(np.unique(y_test)) == K


def test_data_loader_reproductibility():
    K = 7  # nb of classes in "Belle-ile" dataset
    dataset_train_size = 1049
    dataset_test_size = 1049
    N = 45
    p = 13

    bzh_class = Bzh_loader(dataset='unittest',
                           load_npy=False, verbose=False)
    X_train_0, y_train_0 = bzh_class.get_data_train()
    assert len(X_train_0) == dataset_train_size
    assert type(X_train_0) == np.ndarray
    assert X_train_0.shape[1:] == (p, N)
    assert type(y_train_0) == np.ndarray
    assert len(y_train_0) == len(X_train_0)
    assert len(np.unique(y_train_0)) == K

    X_test_0, y_test_0 = bzh_class.get_data_test()
    assert len(X_test_0) == dataset_test_size
    assert type(X_test_0) == np.ndarray
    assert X_test_0.shape[1:] == (p, N)
    assert type(y_test_0) == np.ndarray
    assert len(y_test_0) == len(X_test_0)
    assert len(np.unique(y_test_0)) == K

    bzh_class = Bzh_loader(dataset='unittest',
                           load_npy=False, verbose=False)
    X_train_1, y_train_1 = bzh_class.get_data_train()
    X_test_1, y_test_1 = bzh_class.get_data_test()

    np_testing.assert_allclose(X_train_0, X_train_1)
    np_testing.assert_allclose(y_train_0, y_train_1)
    np_testing.assert_allclose(X_test_0, X_test_1)
    np_testing.assert_allclose(y_test_0, y_test_1)
