import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples, loadMCSamples
from sunbird.inference.samples import Chain
from cosmoprimo.fiducial import AbacusSummit
import numpy as np
from tabulate import tabulate, SEPARATING_LINE

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_samples(statistics, date='apr7', model='LCDM'):
    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/{date}/c000_hod030/{model}/'
    data_fn = Path(data_dir) / f"chain_number_density+{statistics}.npy"
    chain = Chain.load(data_fn)
    samples = Chain.to_getdist(chain, add_derived=False)
    labels = chain.data['labels']
    return samples, labels


params_lcdm = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
params_w0wa = ['w0_fld', 'wa_fld']
params_Nur = ['N_ur']

stats = [
    'wp',
    'minkowski_apr11',
    'tpcf',
    'pk',
    'wst_apr11',
    'bk',
    'dsc_pk',
    'dt_gv',
    # 'minkowski_apr11+wp+tpcf+bk+dsc_pk+wst_apr11+dt_gv',
]
labels = [
    r'Projected 2PCF',
    r'Minkowski functionals',
    r'2PCF multipoles',
    r'Power spectrum multipoles',
    r'Wavelet scattering',
    r'Bispectrum multipoles',
    r'Density-split clustering',
    r'DT voids',
    # r'Greedy combination',
]

# header = ['Statistic', r'$\omega_{\rm cdm}$', '$\sigma_8$']

table = []

table.append([r'$\bm{\Lambda}$\textbf{CDM}'])
for stat, label in zip(stats, labels):
    samples, param_labels = get_samples(stat, date='apr22', model='LCDM')
    constraints = [rf"${samples[param].std():.5f}$" for param in params_lcdm]
    constraints += ['---' for _ in params_w0wa]
    constraints += ['---' for _ in params_Nur]
    table.append([label] + constraints)

table.append([r'$\bm{w_0w_a}$\textbf{CDM}'])

for stat, label in zip(stats, labels):
    samples, param_labels = get_samples(stat, date='apr22', model='w0wa')
    constraints = [rf"${samples[param].std():.5f}$" for param in params_lcdm]
    constraints += [rf"${samples[param].std():.5f}$" for param in params_w0wa]
    constraints += ['---' for _ in params_Nur]
    table.append([label] + constraints)

table.append([r'$\bm{\Lambda}$\textbf{CDM}$+N_{\rm ur}$'])

header = ['Statistic'] + [f'$\Delta$' + param_labels[param] for param in params_lcdm] + \
    [f'$\Delta$' + param_labels[param] for param in params_w0wa] + \
    [f'$\Delta$' + param_labels[param] for param in params_Nur]
print(tabulate(table, tablefmt='latex_raw', headers=header, floatfmt=".5f"))
