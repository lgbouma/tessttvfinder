# -*- coding: utf-8 -*-
'''
are the data better described by a line, or by a quadratic function?

constructed following
http://nbviewer.jupyter.org/url/jakevdp.github.io/downloads/notebooks/FreqBayes5.ipynb
'''

from __future__ import division, print_function

import os, argparse, pickle, h5py
from glob import glob

import matplotlib as mpl
mpl.use('Agg')
import numpy as np, matplotlib.pyplot as plt, pandas as pd

from scipy import stats, optimize, integrate

import emcee

from datetime import datetime

###############################
# initial wrangling functions #
###############################

def get_data():

    data = np.array([[ 0.42,  0.72,  0.  ,  0.3 ,  0.15,
                       0.09,  0.19,  0.35,  0.4 ,  0.54,
                       0.42,  0.69,  0.2 ,  0.88,  0.03,
                       0.67,  0.42,  0.56,  0.14,  0.2  ],
                     [ 0.33,  0.41, -0.22,  0.01, -0.05,
                      -0.05, -0.12,  0.26,  0.29,  0.39,
                       0.31,  0.42, -0.01,  0.58, -0.2 ,
                       0.52,  0.15,  0.32, -0.13, -0.09 ],
                     [ 0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                       0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                       0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                       0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1  ]])

    x, y, sigma_y = data

    return x, y, sigma_y, data


def initial_plot_data(
    x, y, sigma_y, savpath='../results/model_comparison/toy_model/data.png',
    xlabel='x', ylabel='y'):

    fig, ax = plt.subplots()
    ax.errorbar(x, y, sigma_y, fmt='ok', ecolor='gray');
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('data we will fit')

    fig.tight_layout()
    fig.savefig(savpath, bbox_inches='tight')


def plot_maxlikelihood_models(
    x, y, sigma_y, theta_linear, theta_quadratic,
    savpath=os.path.join('../results/model_comparison/toy_model',
                         'data_maxlikelihood_fits.png'),
    xlabel='x', ylabel='y'):

    xfit = np.linspace(0, 1, 1000)

    fig, ax = plt.subplots()

    ax.errorbar(x, y, sigma_y, fmt='ok', ecolor='gray');
    ax.plot(xfit, polynomial_fit(theta_linear, xfit),
            label='best linear model')
    ax.plot(xfit, polynomial_fit(theta_quadratic, xfit),
            label='best quadratic model')

    ax.legend(loc='best', fontsize='x-small')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('data we fit')

    fig.tight_layout()
    fig.savefig(savpath, bbox_inches='tight')



def polynomial_fit(theta, x):
    """Polynomial model of degree (len(theta) - 1)"""

    return sum(t * x ** n for (n, t) in enumerate(theta))


def logL(theta, data, model=polynomial_fit):
    """Gaussian log-likelihood of the model at theta"""

    # unpack the data
    x, y, sigma_y = data

    # evaluate the model at theta
    y_fit = model(theta, x)

    return stats.norm.logpdf(y,y_fit,sigma_y).sum()


def best_theta(degree, data, model=polynomial_fit):
    """Standard frequentist approach: find the model that maximizes the
    likelihood under each model. Here, do it by direct optimization."""

    # create a zero vector of inital values
    theta_0 = np.zeros(degree+1);

    neg_logL = lambda theta: -logL(theta, data, model)

    return optimize.fmin_bfgs(neg_logL, theta_0, disp=False)


#########################
# frequentist functions #
#########################

def compute_chi2(degree, data):

    x, y, sigma_y = data
    theta = best_theta(degree, data)
    resid = (y - polynomial_fit(theta, x)) / sigma_y
    chi2 = np.sum(resid ** 2)

    return chi2


def compute_dof(degree, data):

    return data.shape[1] - (degree + 1)


def chi2_likelihood(degree, data):

    chi2 = compute_chi2(degree, data)
    dof = compute_dof(degree, data)

    print('degree: {:d}\tchi2: {:.6f}\tdof: {:d}'
          .format(degree, chi2, int(dof)))

    return stats.chi2(dof).pdf(chi2)


def plot_chi2_diff_distribution_comparison(
    data,
    savpath=(
        os.path.join('../results/model_comparison/toy_model',
                     'chi2_diff_distribution_comparison.png'))
    ):

    chi2_diff = compute_chi2(1, data) - compute_chi2(2, data)

    # The p value in this context means that, assuming the linear model is
    # true, there is a 17% probability that simply by chance we would see data
    # that favors the quadratic model more strongly than the data we have.
    v = np.linspace(1e-3, 5, 1000)
    chi2_dist = stats.chi2(1).pdf(v)
    # Calculate p value through survival function of the chi2 distribution.
    p_value = stats.chi2(1).sf(chi2_diff)

    fig, ax = plt.subplots()
    ax.fill_between(v, 0, chi2_dist, alpha=0.3)
    ax.fill_between(v, 0, chi2_dist * (v > chi2_diff), alpha=0.5)
    ax.axvline(chi2_diff)

    ax.set_ylim((0, 1))
    ax.set_xlabel("$\chi^2$ difference")
    ax.set_ylabel("probability")

    ax.text(0.97, 0.97, "p = {0:.2f}".format(p_value),
            ha='right', va='top', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(savpath, bbox_inches='tight')

######################
# bayesian functions #
######################

def log_prior(theta):
    # size of theta determines the model.
    # flat prior over a large range [-100,100]
    if np.any(abs(theta) > 100):
        return -np.inf  # log(0)
    else:
        return -len(theta)*np.log(200);


def log_likelihood(theta, data):

    x, y, sigma_y = data
    yM = polynomial_fit(theta, x)

    return -0.5 * np.sum(np.log(2 * np.pi * sigma_y ** 2)
                         + (y - yM) ** 2 / sigma_y ** 2)


def log_posterior(theta, data):

    theta = np.asarray(theta)

    return log_prior(theta) + log_likelihood(theta, data)


def integrate_posterior_2D(posterior, xlim, ylim, data, logprobs=True):

    if(logprobs):
        func = (
            lambda theta1, theta0:
            np.exp(log_posterior([theta0, theta1], data))
        )

    else:
        func = (
            lambda theta1, theta0: posterior([theta0, theta1], data)
        )

    return integrate.dblquad(func, xlim[0], xlim[1],
                             lambda x: ylim[0], lambda x: ylim[1],
                             epsabs=1e-1)


def integrate_posterior_3D(log_posterior, xlim, ylim, zlim, data,
                           logprobs=True):

    if(logprobs):
        func = (
            lambda theta2, theta1, theta0:
            np.exp(log_posterior([theta0, theta1, theta2], data))
        )

    else:
        func = (
            lambda theta2, theta1, theta0:
            posterior([theta0, theta1, theta2], data)
        )

    return integrate.tplquad(
        func, xlim[0], xlim[1],
        lambda x: ylim[0], lambda x: ylim[1],
        lambda x, y: zlim[0], lambda x, y: zlim[1],
        epsabs=1e-1
    )


def compute_mcmc(degree, data,
                 log_posterior=log_posterior,
                 nwalkers=50, nburn=1000, nsteps=2000):

    ndim = degree + 1  # this determines the model
    rng = np.random.RandomState(0)
    starting_guesses = rng.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data])
    sampler.run_mcmc(starting_guesses, nsteps)
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    return trace


def plot_mcmc_posteriors(
    trace_2D, trace_3D
    savpath_2D='../results/model_comparison/toy_model/samples_linear.png',
    savpath_3D='../results/model_comparison/toy_model/samples_quadratic.png'):

    import seaborn as sns

    columns = [r'$\theta_{0}$'.format(i) for i in range(3)]
    df_2D = pd.DataFrame(trace_2D, columns=columns[:2])

    with sns.axes_style('ticks'):
        plt.close("all")
	jointplot = sns.jointplot(r'$\theta_0$', r'$\theta_1$',
                           data=df_2D, kind="hex")
        plt.savefig(savpath_2D)

	df_3D = pd.DataFrame(trace_3D, columns=columns[:3])

	# get the colormap from the joint plot above
	cmap = jointplot.ax_joint.collections[0].get_cmap()

	with sns.axes_style('ticks'):
            plt.close("all")
            grid = sns.PairGrid(df_3D)
            grid.map_diag(plt.hist, bins=30, alpha=0.5)
            grid.map_offdiag(plt.hexbin, gridsize=50, linewidths=0, cmap=cmap)
            plt.savefig(savpath_3D)


#################################
# main model comparison routine #
#################################
def main():

    x, y, sigma_y, data = get_data()
    initial_plot_data(x, y, sigma_y)

    # Look at maximum likelihood models for linear and quadratic case.
    theta_linear = best_theta(1, data)
    theta_quadratic = best_theta(2, data)
    plot_maxlikelihood_models(x, y, sigma_y, theta_linear, theta_quadratic)

    # Frequentist approach: treat the linear model as the null hypothesis. Ask:
    # is there enough evidence to justify a more complicated quadratic model?
    # Answer by comparing the observed chi^2 difference to its expected
    # distribution,
    #    f(x, df) = \frac{1}{(2 \gamma(df/2)} (x/2)^{df/2-1} \exp(-x/2).
    # (See docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)

    print('frequentist approach')

    chi2_likelihood_linear = chi2_likelihood(1, data)
    chi2_likelihood_quadratic = chi2_likelihood(2, data)

    for ix, chi2_like in enumerate(
        [chi2_likelihood_linear, chi2_likelihood_quadratic]):

        degree = ix+1
        print('degree: {:d}\tchi2 likelihood: {:.6f}'
              .format(degree, chi2_like))

    plot_chi2_diff_distribution_comparison(data)

    # Bayesian approach: compute the _odds ratios_ between the two models:
    # 
    #   OR(M_1,M_2) = P(M_2 | D) / P(M_1 | D)
    #               = P(D | M_2) P(M_2) / (P(D | M_1) P(M_1)).
    #
    # Take the prior odds ratio, P(M_2)/P(M_1), to be = 1.
    # 
    # We then want the ratio of marginal model likelihoods, AKA the Bayes
    # factor:
    #
    #       bayes factor = P(D | M_2) / P(D | M_1).
    # 
    # We can compute it by noting:
    #
    #   P(D|M) = int_{Θ} P(D|θ,M) P(θ|M) dθ,
    #
    # by the definition of conditional probabilities. This can be
    # computationally intensive, but these models have not many parameters.

    # first, solve the model fitting problem. what are the best-fitting model
    # parameters for each model?
    trace_2D = compute_mcmc(1, data)
    trace_3D = compute_mcmc(2, data)

    plot_mcmc_posteriors(trace_2D, trace_3D)

    # NOTE: chosing integration limit is tricky. Arbitrarily can set them based
    # on what is known about the problem. Perhaps do it by generating samples
    # from posterior to understand what range of values are possible.
    # NOTE: you also need to choose the priors!
    xlim, ylim, zlim = (-5,5), (-5,5), (-5,5)

    # Integrate over the two parameters in the linear fit to get the posterior
    # probability for the linear model.
    Z_linear, err_Z_linear = (
        integrate_posterior_2D(log_posterior, xlim, ylim, data)
    )

    # Integrate over the three parameters in the quadratic fit to get the posterior
    # probability for the quadratic model.
    # (Takes ~5 minutes or so to do this integral.)
    print('{:s}: beginning big integral'.format(datetime.utcnow().isoformat()))
    Z_quadratic, err_Z_quadratic = (
        integrate_posterior_3D(log_posterior, xlim, ylim, zlim, data)
    )
    print('{:s}: finished'.format(datetime.utcnow().isoformat()))

    bayes_factor = Z_quadratic/Z_linear

    print('bayesian approach')
    print('bayes factor = Z_quadratic/Z_linear = {:.4e}'.format(bayes_factor))


if __name__ == "__main__":
    main()
