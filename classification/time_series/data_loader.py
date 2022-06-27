import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
import breizhcrops as bzh
from copy import deepcopy
import os
import sys
from tqdm import tqdm


class Bzh_loader():
    def __init__(self, seed=0, dataset='train_test',
                 path_npy='breizhcrops_npy', load_npy=False,
                 level='L1C', verbose=True):
        self.dataset = dataset
        self.level = level
        self.verbose = verbose
        self.classname = None
        self.path_npy = path_npy

        list_available_datasets = self.get_list_available_datasets()

        if dataset not in list_available_datasets:
            raise ValueError('Dataset ' + str(dataset) + ' does not exist...')

        if verbose:
            print('Load dataset...')

        rnd.seed(seed)
        if load_npy:
            self._load_npy()
        else:
            self._load()

    @staticmethod
    def get_list_available_datasets():
        list_available_datasets = ['train_test', 'train_small_test']
        list_available_datasets += ['train_val', 'train_val_small']
        list_available_datasets += ['unittest', 'unittest_small']
        return list_available_datasets

    def _load_list_datasets(self, list_datasets, level):
        verbose = self.verbose

        preload_ram = True

        X_total = None
        y_total = None

        for dataset_name in list_datasets:
            if verbose:
                print('Load dataset:', dataset_name)
            else:
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')

            dataset = bzh.BreizhCrops(
                dataset_name,
                level=level,
                preload_ram=preload_ram,
                verbose=verbose
            )

            if not verbose:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

            if self.classname is None:
                if dataset_name == 'belle-ile':
                    tmp = [str(k) for k in range(7)]
                else:
                    tmp = dataset.classname
                self.classname = tmp

            N, p = dataset[0][0].shape

            indices = list(range(len(dataset)))
            dataset_size = len(indices)
            X, y = np.zeros((dataset_size, p, N)), np.zeros(dataset_size)

            iterator = indices
            if verbose:
                iterator = tqdm(indices)

            for i, idx in enumerate(iterator):
                temp = dataset[idx][0].T
                assert temp.shape == (p, N)
                X[i] = temp
                y[i] = int(dataset[idx][1])

            if X_total is None:
                X_total = X
            else:
                X_total = np.concatenate([X_total, X])

            if y_total is None:
                y_total = y
            else:
                y_total = np.concatenate([y_total, y])

        return X_total, y_total

    def _load(self):
        verbose = self.verbose
        dataset = self.dataset
        if dataset in ['train_test', 'train_small_test']:
            dataset_train = ['frh01', 'frh02', 'frh03']
            dataset_test = ['frh04']
        elif dataset in ['train_val', 'train_val_small']:
            dataset_train = ['frh01', 'frh02']
            dataset_test = ['frh03']
        elif dataset in ['unittest', 'unittest_small']:
            dataset_train = ['belle-ile']
        else:
            raise ValueError('Wrong dataset...')

        # load training data
        X_train, y_train = self._load_list_datasets(
            list_datasets=dataset_train,
            level=self.level
        )

        # load test data
        if dataset in ['unittest', 'unittest_small']:
            X_test = deepcopy(X_train)
            y_test = deepcopy(y_train)
        else:
            X_test, y_test = self._load_list_datasets(
                list_datasets=dataset_test,
                level=self.level
            )

        # center data points
        mean = np.mean(X_train, axis=(0, 2))
        mean = mean[:, np.newaxis]
        X_train = X_train - mean
        X_test = X_test - mean

        # check and print mean of X_train and X_test
        mean = np.mean(X_train, axis=(0, 2))
        assert la.norm(mean) < 1e-8
        if verbose:
            print('Norm of mean of train set:', la.norm(mean))
        mean = np.mean(X_test, axis=(0, 2))
        if verbose:
            print('Norm of mean of val/test set:', la.norm(mean))

        # sample data to get a small subset of training set
        str_datasets = ['train_small_test', 'train_val_small',
                        'unittest_small']
        if dataset in str_datasets:
            THRESHOLD_N_SAMPLES_TRAIN = 1000

            X_train_new, y_train_new = list(), list()
            for k in np.unique(y_train):
                mask_k = (y_train == k)
                if np.sum(mask_k) < THRESHOLD_N_SAMPLES_TRAIN:
                    X_train_new.append(X_train[mask_k])
                    y_train_new.append(y_train[mask_k])
                else:
                    index = rnd.choice(
                        X_train[mask_k].shape[0],
                        THRESHOLD_N_SAMPLES_TRAIN,
                        replace=False
                    )
                    X_train_new.append(X_train[mask_k][index])
                    y_train_new.append(y_train[mask_k][index])
            X_train = np.concatenate(X_train_new)
            y_train = np.concatenate(y_train_new)

        # sample data to get a small subset of test set
        str_datasets = ['train_val_small', 'unittest_small']
        if dataset in str_datasets:
            THRESHOLD_N_SAMPLES_TEST = 10
            FACTOR_TEST = 0.1

            X_test_new, y_test_new = list(), list()
            for k in np.unique(y_test):
                mask_k = (y_test == k)
                if np.sum(mask_k) < THRESHOLD_N_SAMPLES_TEST:
                    X_test_new.append(X_test[mask_k])
                    y_test_new.append(y_test[mask_k])
                else:
                    n = int(np.sum(mask_k) * FACTOR_TEST)
                    index = rnd.choice(
                        X_test[mask_k].shape[0], n, replace=False)
                    X_test_new.append(X_test[mask_k][index])
                    y_test_new.append(y_test[mask_k][index])
            X_test = np.concatenate(X_test_new)
            y_test = np.concatenate(y_test_new)

        # set data
        self._set_data(X_train, y_train, X_test, y_test)

    def _path_classname(self):
        dataset = self.dataset
        level = self.level
        path_npy = self.path_npy
        filename = 'classname_' + dataset + '_' + level + '.npy'
        path = os.path.join(path_npy, filename)
        return path

    def _path_npy_train(self):
        dataset = self.dataset
        level = self.level
        path_npy = self.path_npy
        filename = 'X_train_' + dataset + '_' + level + '.npy'
        path_X_train = os.path.join(path_npy, filename)
        filename = 'y_train_' + dataset + '_' + level + '.npy'
        path_y_train = os.path.join(path_npy, filename)
        return path_X_train, path_y_train

    def _path_npy_test(self):
        dataset = self.dataset
        level = self.level
        path_npy = self.path_npy
        filename = 'X_test_' + dataset + '_' + level + '.npy'
        path_X_test = os.path.join(path_npy, filename)
        filename = 'y_test_' + dataset + '_' + level + '.npy'
        path_y_test = os.path.join(path_npy, filename)
        return path_X_test, path_y_test

    def save_into_npy(self):
        path_npy = self.path_npy
        if not os.path.isdir(path_npy):
            os.mkdir(path_npy)

        # save classname
        classname = self.classname
        path_classname = self._path_classname()
        np.save(path_classname, classname)

        # get training data
        X_train, y_train = self.get_data_train()

        # save training data
        path_X_train, path_y_train = self._path_npy_train()
        np.save(path_X_train, X_train)
        np.save(path_y_train, y_train)

        # get test data
        X_test, y_test = self.get_data_test()

        # save test data
        path_X_test, path_y_test = self._path_npy_test()
        np.save(path_X_test, X_test)
        np.save(path_y_test, y_test)

    def _load_npy(self):
        # save classname
        path_classname = self._path_classname()
        classname = np.load(path_classname, allow_pickle=True)
        self.classname = classname

        # load training data
        path_X_train, path_y_train = self._path_npy_train()
        X_train = np.load(path_X_train, allow_pickle=True)
        y_train = np.load(path_y_train, allow_pickle=True)

        # load test data
        path_X_test, path_y_test = self._path_npy_test()
        X_test = np.load(path_X_test, allow_pickle=True)
        y_test = np.load(path_y_test, allow_pickle=True)

        # set data
        self._set_data(X_train, y_train, X_test, y_test)

    def _set_data(self, X_train, y_train, X_test, y_test):
        assert X_train.shape[1:] == X_test.shape[1:]
        assert (np.unique(y_train) == np.unique(y_test)).all()

        p, N = X_test.shape[1:]
        K = len(np.unique(y_test))

        self.p = p
        self.N = N
        self.K = K

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

    def get_data_train(self):
        return self.X_train, self.y_train

    def get_data_test(self):
        return self.X_test, self.y_test

    def get_info(self):
        return self.p, self.N, self.K

    def print_info(self):
        dataset = self.dataset
        classname = self.classname
        p, N, K = self.get_info()
        X_train, y_train = self.get_data_train()
        X_test, y_test = self.get_data_test()
        print()
        print('Dataset:', dataset)
        print('p:', p)
        print('Length of time series:', N)
        print('Number of classes:', K)
        print()
        print('X_train.shape:', X_train.shape)
        print('X_test.shape:', X_test.shape)
        print()
        if y_train is not None:
            print('Train set classes:', len(y_train), 'samples.')
            print()
            for k in range(len(classname)):
                print(classname[k], ':',
                      int(np.sum(y_train == k)), 'samples.')
            print()
        print('Test set classes:', len(y_test), 'samples.')
        print()
        for k in range(len(classname)):
            print(classname[k], ':', int(np.sum(y_test == k)), 'samples.')
        print()
