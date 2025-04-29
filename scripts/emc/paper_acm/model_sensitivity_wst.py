from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference.priors import Yuan23, AbacusSummit
from sunbird import setup_logging

import acm.observables.emc as emc

from pathlib import Path
import numpy as np
import argparse

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_priors(cosmo=True, hod=True):
    stats_module = 'scipy.stats'
    priors, ranges, labels = {}, {}, {}
    if cosmo:
        priors.update(AbacusSummit(stats_module).priors)
        ranges.update(AbacusSummit(stats_module).ranges)
        labels.update(AbacusSummit(stats_module).labels)
    if hod:
        priors.update(Yuan23(stats_module).priors)
        ranges.update(Yuan23(stats_module).ranges)
        labels.update(Yuan23(stats_module).labels)
    return priors, ranges, labels


parser = argparse.ArgumentParser()
parser.add_argument("--cosmo_idx", type=int, default=0)
parser.add_argument("--hod_idx", type=int, default=30)

args = parser.parse_args()
setup_logging()

# set up the inference
priors, ranges, labels = get_priors(cosmo=True, hod=True)

# load observables with their custom filters
observable = emc.WaveletScatteringTransform(
    select_mocks={
        'cosmo_idx': args.cosmo_idx, 'hod_idx': args.hod_idx,
    },
    select_coordinates={'multipoles': [0, 2]},
)

statistics = observable.stat_name

# load the data
data_x = observable.lhc_x
data_x_names = observable.lhc_x_names
data_y = observable.lhc_y
print(f'Loaded LHC x with shape: {data_x.shape}')
print(f'Loaded LHC y with shape {data_y.shape}')

# load the covariance matrix
covariance_matrix = observable.get_covariance_matrix(divide_factor=64)
error = np.sqrt(np.diag(covariance_matrix))
print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

s = observable.separation


for param_name in ['wa_fld']:

    fig, ax = plt.subplots(1, figsize=(10, 5))

    prior_sample = np.linspace(ranges[param_name][0], ranges[param_name][1], 100)
    prior_norm = prior_sample - prior_sample.min()
    prior_norm /= prior_norm.max()

    cmap = matplotlib.cm.get_cmap('RdBu')

    for i, param_value in enumerate(prior_sample):
        param_idx = data_x_names.index(param_name)
        data_x[param_idx] = param_value
        pred_y = observable.get_model_prediction(data_x)

        ax.plot(pred_y[:len(s)], color=cmap(prior_norm[i]))

    divider = make_axes_locatable(fig.axes[0])
    cax = divider.append_axes('top', size="7%", pad=0.15)

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=prior_sample.min(), vmax=prior_sample.max()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1.2, vmax=-0.8))
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label(labels[param_name], rotation=0, labelpad=10, fontsize=20)

    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    ax.tick_params(axis='both', which='major', labelsize=11)

    ax.set_ylabel(r'$\textrm{WST coefficient}$', fontsize=13)
    ax.set_xlabel(r'$\textrm{bin index}$', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    plt.savefig(f'model_sensitivity_wst_{param_name}.png', dpi=300, bbox_inches='tight')