import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from acm.data.io_tools import *
import acm.observables.emc as emc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_data(statistic, return_model=True):
    stat = getattr(emc, statistic)
    observable = stat(
        select_mocks={'cosmo_idx': 0, 'hod_idx': 30},
        select_coordinates=select_coordinates,
        slice_coordinates=slice_coordinates
    )
    data_y = observable.lhc_y
    data_x = observable.lhc_x
    covariance_matrix = observable.get_covariance_matrix(divide_factor=64)
    emulator_error = observable.get_emulator_error()
    data_error = np.sqrt(np.diag(covariance_matrix))
    model = observable.get_model_prediction(data_x)
    sep = observable.separation[:data_y.shape[0]]
    return sep, emulator_error, data_error, data_y, model


slice_coordinates = {}

fig, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

# # projected correlation function
# statistic = 'GalaxyProjectedCorrelationFunction'
# select_coordinates = {}
# sep, emulator_error, data_error = get_data(statistic)
# ax[0, 0].plot(sep, emulator_error/data_error)
# # ax[0, 0].set_xscale('log')
# ax[0][0].set_xlabel(r'$r\,[h^{-1}{\rm Mpc}]$', fontsize=15)
# ax[0][0].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
# ax[0][0].set_title(r'$\textrm{Projected 2PCF}$', fontsize=15)

# 2PCF multipoles
statistic = 'GalaxyCorrelationFunctionMultipoles'
select_coordinates = {'multipoles': [0]}
slice_coordinates = {'s': [0, 50]}
s, emulator_error, data_error, data_y, model_y = get_data(statistic)
ax[0].errorbar(s, s**2 * data_y, s ** 2 * data_error, label='simulation', ls='',
    marker='o', ms=3.0, elinewidth=1.0, markeredgewidth=0.5, capsize=2, mfc="None", color='dimgrey')
ax[0].plot(s, s**2 * model_y, label='emulator', linewidth=1.0, color='hotpink')
ax[1].plot(s, data_error/data_y, label='data precision', color='dimgrey')
ax[1].plot(s, emulator_error/data_y, ls='--', label='emulator precision', color='hotpink')
ax[0].legend(fontsize=10, handlelength=1.25)
ax[1].legend(fontsize=10, handlelength=1.25)
ax[1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=15)
ax[1].set_ylabel(r'$\sigma\xi/\xi$', fontsize=15)
ax[0].set_ylabel(r'$s^2\xi_0(s)\,[h^{-2}{\rm Mpc^2}]$', fontsize=15)

# # power spectrum
# statistic = 'GalaxyPowerSpectrumMultipoles'
# for ell in [0, 2]:
#     select_coordinates = {'multipoles': [ell]}
#     sep, emulator_error, data_error = get_data(statistic)
#     ax[0, 2].plot(sep, emulator_error/data_error, label=f'$\ell={ell}$')

# ax[0][2].set_xlabel(r'$k\,[h/{\rm Mpc}]$', fontsize=15)
# ax[0][2].set_ylabel(r'$(X_{\rm model} - X_{\rm data})/\sigma_{\rm data}$', fontsize=15)
# ax[0][2].set_title(r'$\textrm{Power spectrum multipoles}$', fontsize=15)
# ax[0][2].legend(fontsize=13)

plt.tight_layout()
# plt.savefig('emulator_error_multipanel.pdf', bbox_inches='tight')
plt.savefig('tpcf_understand_error.png', bbox_inches='tight', dpi=300)