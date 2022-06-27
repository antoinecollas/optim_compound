from classification.generate_npy import main


def test_generate_npy():
    PATH_NPY = 'tests_breizhcrops_npy'
    STR_DATASETS = ['unittest', 'unittest_small']

    main(
        seed=0,
        path_npy=PATH_NPY,
        str_datasets=STR_DATASETS,
        level='L1C',
        verbose=False
    )
