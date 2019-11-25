#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os

def standard_cable(theta, theta_wp, theta_fc):
    beta = (theta - theta_wp) / (theta_fc - theta_wp)
    beta = np.where(beta > 1.0, 1.0, beta)
    beta = np.where(beta < 0.0, 0.0, beta)
    return (beta)

def fsig_tuzet(psi_leaf, sf, psi_f):
    """
    An empirical logistic function to describe the sensitivity of stomata
    to leaf water potential.

    Sigmoid function assumes that stomata are insensitive to psi_leaf at
    values close to zero and that stomata rapidly close with decreasing
    psi_leaf.

    Parameters:
    -----------
    psi_leaf : float
        leaf water potential (MPa)

    Returns:
    -------
    fw : float
        sensitivity of stomata to leaf water potential [0-1]

    Reference:
    ----------
    * Tuzet et al. (2003) A coupled model of stomatal conductance,
      photosynthesis and transpiration. Plant, Cell and Environment 26,
      10971116

    """
    num = 1.0 + np.exp(sf * psi_f)
    den = 1.0 + np.exp(sf * (psi_f - psi_leaf))
    fw = num / den

    return fw


def fsig_hydr(psi_stem, p50, s50):
    """
    Calculate the relative conductance as a function of xylem pressure
    using the Weibull (sigmoidal) model based on values of P50
    and S50 which are obtained by fitting curves to measured data.

    Higher values for s50 indicate a steeper response to xylem pressure.

    Parameters:
    -----------
    psi_stem : object
        stem water potential, MPa

    Returns:
    --------
    relk : float
        relative conductance (K/Kmax) as a funcion of xylem pressure (-)

    References:
    -----------
    * Duursma & Choat (2017). Journal of Plant Hydraulics, 4, e002.
    """
    # xylem pressure
    P = np.abs(psi_stem)

    # the xylem pressure (P) x% of the conductivity is lost
    PX = np.abs(p50)
    V = (50.0 - 100.) * np.log(1.0 - 50. / 100.)
    p = (P / PX)**((PX * s50) / V)

    # relative conductance (K/Kmax) as a funcion of xylem pressure
    relk = (1. - 50. / 100.)**p

    return (relk)


if __name__ == "__main__":

    # WSF params
    p50 = -3.002384
    s50 =  35.26948
    sf = 2.000000
    psi_f = -2.455474
    psi = np.linspace(-0.05, -5., 20)
    theta_fc = [0.143,0.301,0.367,0.218,0.310,0.370,0.255]
    theta_wp = [0.072,0.216,0.286,0.135,0.219,0.283,0.175]
    sand_idx = 0
    theta = np.linspace(0.18, 0.025, 100)
    beta_cable = standard_cable(theta, theta_wp[sand_idx], theta_fc[sand_idx])
    beta_leaf = fsig_tuzet(psi, sf, psi_f)
    beta_stem = fsig_hydr(psi, p50, s50)


    fig = plt.figure(figsize=(16,6))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(theta, beta_cable, label="$\Psi$$_{l}$", color="black")
    ax1.set_xlabel("Volumetric soil water content (m$^{3}$ m$^{-3}$)")
    ax1.set_ylabel("Water stress modifier (-)")

    ax2.plot(psi, beta_leaf, label="$\Psi$$_{l}$", ls="-", color="royalblue")
    ax2.plot(psi, beta_stem, label="$\Psi$$_{x}$", ls="-", color="seagreen")
    ax2.set_xlabel("Water potential (MPa)")
    ax2.legend(numpoints=1, loc=(0.03, 0.8), frameon=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)


    plt.show()
    odir = "/Users/mdekauwe/Dropbox/Drought_risk_paper/figures/figs"
    fig.savefig(os.path.join(odir, "drought_modifers.pdf"), bbox_inches='tight',
                pad_inches=0.1)
