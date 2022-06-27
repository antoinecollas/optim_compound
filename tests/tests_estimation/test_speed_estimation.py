from estimation.speed_comparison import main


def test_speed_comparison():
    p = 10
    main(
        N=10*p,
        p=p,
        nu=1,
        reg_types=['L1', 'L2', 'BW', 'KL'],
        reg_betas=[0, 1e-6, 1e-4, 1e-2, 1],
        iter_max=3,
        solvers=['steepest', 'conjugate'],
        solvers_IG=['steepest'],
        verbose=False
    )
