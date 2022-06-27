import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
import numpy.testing as np_test
from pyCovariance.features.base import _FeatureArray
from pyCovariance.manifolds import ComplexCompoundGaussianIGConstrainedTexture
from pyCovariance.generation_data import\
        generate_covariance,\
        generate_textures_gamma_dist

from center_of_mass.computation.computation import\
        _create_cost_egrad,\
        estimate_center_of_mass_constrained_texture


def _rand(p, N):
    theta = list()
    theta.append(rnd.normal(size=(p, 1)))
    theta.append(generate_covariance(p))
    theta.append(generate_textures_gamma_dist(N, unit_prod=True))
    return theta


def _check_dim_type(theta, p, N):
    mu, sigma, tau = theta

    assert type(mu) == np.ndarray
    assert mu.dtype == np.float64
    assert mu.shape == (p, 1)

    assert type(sigma) == np.ndarray
    assert sigma.dtype == np.float64
    assert sigma.shape == (p, p)
    np_test.assert_allclose(sigma, sigma.T)
    eig = la.eigvalsh(sigma)
    assert (eig > 0).all()

    assert type(tau) == np.ndarray
    assert tau.dtype == np.float64
    assert tau.shape == (N, 1)
    assert (tau > 0).all()
    np_test.assert_allclose(np.prod(tau), 1)


def test__create_cost_egrad():
    p = 5
    N = 10
    M = 3

    # manifold
    man = ComplexCompoundGaussianIGConstrainedTexture(p=p, n=N, k=1, alpha=0)

    # generate data
    X = _FeatureArray((p, 1), (p, p), (N, 1))
    for i in range(M):
        X.append(_rand(p, N))
    theta = _rand(p, N)

    # cost fct
    cost, _ = _create_cost_egrad(X.export())

    # true value
    true_value = 0
    for i in range(M):
        true_value += man.div_alpha_real_case(theta, X[i].export())**2
        true_value += man.div_alpha_real_case(X[i].export(), theta)**2
    true_value /= 2*M

    np_test.assert_allclose(cost(theta[0], theta[1], theta[2]), true_value)


def test_estimate_center_of_mass_constrained_texture():
    p = 5
    N = 10
    M = 3

    # generate data
    X = _FeatureArray((p, 1), (p, p), (N, 1))
    for i in range(M):
        X.append(_rand(p, N))

    # info geo
    mu, sigma, tau, _ = estimate_center_of_mass_constrained_texture(
        X, information_geometry=True, solver='steepest')
    _check_dim_type([mu, sigma, tau], p, N)

    # decoupled metric
    mu, sigma, tau, _ = estimate_center_of_mass_constrained_texture(
        X, information_geometry=False, solver='steepest')
    _check_dim_type([mu, sigma, tau], p, N)
