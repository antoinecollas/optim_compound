from tqdm import tqdm

from classification.time_series import Bzh_loader


def main(seed, path_npy, str_datasets, level, verbose=True):
    if verbose:
        print('seed:', seed)
        print()

    iterator = str_datasets
    if verbose:
        iterator = tqdm(iterator)

    for dataset in iterator:
        if verbose:
            print('dataset:', dataset)

        bzh = Bzh_loader(
            seed=seed,
            dataset=dataset,
            path_npy=path_npy,
            level=level,
            verbose=verbose
        )
        bzh.save_into_npy()


if __name__ == '__main__':
    SEED = 0
    PATH_NPY = 'breizhcrops_npy'
    STR_DATASETS = Bzh_loader.get_list_available_datasets()
    LEVEL = 'L1C'

    main(
        seed=SEED,
        path_npy=PATH_NPY,
        str_datasets=STR_DATASETS,
        level=LEVEL
    )
