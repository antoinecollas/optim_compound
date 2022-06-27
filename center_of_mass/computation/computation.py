import autograd
import autograd.numpy as np
from pyCovariance.features.base import _FeatureArray
from pyCovariance.manifolds import\
        ComplexCompoundGaussianIGConstrainedTexture,\
        SpecialStrictlyPositiveVectors
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import\
        ComplexEuclidean,\
        HermitianPositiveDefinite,\
        Product
from pymanopt.solvers import ConjugateGradient, SteepestDescent


def _get_dim(X):
    assert len(X[0].shape) == 3
    assert len(X[1].shape) == 3
    assert len(X[2].shape) == 3
    M = X[0].shape[0]  # number of location-covariance-texture
    p = X[0].shape[1]  # dimension
    N = X[2].shape[1]  # number of textures
    assert X[0].shape == (M, p, 1)  # location
    assert X[1].shape == (M, p, p)  # covariance
    assert X[2].shape == (M, N, 1)  # texture
    return M, p, N


def _create_cost_egrad(X):
    M, p, N = _get_dim(X)
    man = ComplexCompoundGaussianIGConstrainedTexture(p=p, n=N, k=M, alpha=0)

    @pymanopt.function.Callable
    def _cost(mu, sigma, tau):
        theta = [mu, sigma, tau]
        theta_batch = [
            np.tile(
                theta[i],
                reps=(M, 1, 1)
            )
            for i in range(len(theta))
        ]
        d_squared = man.div_alpha_real_case(theta_batch, X)**2
        d_squared = d_squared + (man.div_alpha_real_case(X, theta_batch)**2)
        d_squared = (1/(2*M)) * d_squared
        return d_squared

    @pymanopt.function.Callable
    def _egrad(mu, sigma, tau):
        egrad = autograd.grad(_cost, argnum=[0, 1, 2])(mu, sigma, tau)
        return egrad

    return _cost, _egrad


def estimate_center_of_mass_constrained_texture(
    X,
    information_geometry=True,
    min_grad_norm=1e-4,
    min_step_size=1e-8,
    iter_max=500,
    time_max=np.inf,
    solver='steepest',
):
    """ A function that estimates the center of mass of
    location-covariance-texture features
    with a constraint of unitary product on the textures.
        Inputs:
            * X = a _FeatureArray containing the
            location-covariance-texture
            * init = point on manifold to initialize the computation
            * information_geometry = use manifold of Compound distribution
            * min_grad_norm = minimum norm of gradient
            * min_step_size = minimum step size
            * iter_max = maximum number of iterations
            * time_max = maximum time in seconds
            * solver = steepest or conjugate
        Outputs:
            * mu = location
            * sigma = covariance
            * tau = textures
            * log = informations about the optimization"""
    if not (type(X) is _FeatureArray):
        raise TypeError('X variable should be a _FeatureArray')
    X = X.export()

    M, p, N = _get_dim(X)

    # Initialisation
    init = list()
    init.append(np.mean(X[0], axis=0))
    init.append(np.mean(X[1], axis=0))
    tmp = np.mean(X[2], axis=0)
    tmp = tmp / np.exp(np.mean(np.log(tmp)))
    init.append(tmp)

    # cost, egrad
    cost, egrad = _create_cost_egrad(X)

    # solver
    if solver == 'steepest':
        solver = SteepestDescent
    elif solver == 'conjugate':
        solver = ConjugateGradient
    else:
        s = 'Solvers available: steepest, conjugate.'
        raise ValueError(s)
    solver = solver(maxtime=time_max, maxiter=iter_max,
                    mingradnorm=min_grad_norm, minstepsize=min_step_size,
                    maxcostevals=np.inf, logverbosity=2)

    # manifold
    if information_geometry:
        manifold = ComplexCompoundGaussianIGConstrainedTexture(
            p=p, n=N, alpha=0)
    else:
        manifold = Product([ComplexEuclidean(p, 1),
                            HermitianPositiveDefinite(p),
                            SpecialStrictlyPositiveVectors(N)])

    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)
    Xopt, log = solver.solve(problem, x=init)

    return Xopt[0], Xopt[1], Xopt[2], log
