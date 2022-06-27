from center_of_mass.speed_comparison import main


def test_speed_comparison():
    main(
        Ms=[2],
        N=4,
        p=3,
        nu=10,
        iter_max=100,
        solvers=['steepest'],
        solvers_IG=['steepest'],
        verbose=False
    )
