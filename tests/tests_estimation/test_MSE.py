from estimation.MSE import main


def test_MSE():
    n_MC = 3
    p = 4
    nu = 1
    list_n_samples = [30, 100]
    iter_max = 50
    min_grad_norm = 1e-5
    min_step_size = 1e-10,
    solvers_IG = ['steepest']
    n_jobs = 1
    verbose = False

    main(
        n_MC=n_MC,
        p=p,
        nu=nu,
        list_n_samples=list_n_samples,
        iter_max=iter_max,
        min_grad_norm=min_grad_norm,
        min_step_size=min_step_size,
        solvers_IG=solvers_IG,
        scatter_use_SPD_dist=False,
        n_jobs=n_jobs,
        verbose=verbose
    )

    main(
        n_MC=n_MC,
        p=p,
        nu=nu,
        list_n_samples=list_n_samples,
        iter_max=iter_max,
        min_grad_norm=min_grad_norm,
        min_step_size=min_step_size,
        solvers_IG=solvers_IG,
        scatter_use_SPD_dist=True,
        n_jobs=n_jobs,
        verbose=verbose
    )
