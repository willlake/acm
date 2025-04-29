import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples, loadMCSamples
from sunbird.inference.samples import Chain
from cosmoprimo.fiducial import AbacusSummit
import numpy as np
import glob

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


chains = []
legend_labels = []
color_dict = {'c000': 'dimgrey', 'c001': 'orange', 'c002': 'dodgerblue', 'c003': 'forestgreen', 'c004': 'crimson'}
marker_dict = {'c000': 'o', 'c001': 's', 'c002': '^', 'c003': 'D', 'c004': 'X'}

# params = ['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
# params = ['omega_cdm', 'sigma8_m', 'n_s', 'logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
params = ['omega_cdm', 'sigma8_m']

offsets = []
means = []
colors = []
markers = []

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/optimal/apr11/c00?_hod*/LCDM/'
# data_fns = Path(data_dir) / 'chain_number_density+minkowski_apr11+wp+tpcf+bk+dsc_pk+wst_apr11+dt_gv.npy'
data_fns = Path(data_dir) / 'chain_number_density+minkowski_apr11+wp+tpcf+pk+bk+dsc_pk+wst_apr11+dt_gv+voxel_voids+pdf_r10+cgf_r10.npy'
for data_fn in glob.glob(str(data_fns)):
    cosmo = data_fn.split('apr11/')[1][:4]
    chain = Chain.load(data_fn)
    samples = Chain.to_getdist(chain, add_derived=True)
    mean = samples.mean(params)
    fid_cosmo = AbacusSummit(int(cosmo[1:]))
    chain.markers.update({'Omega_m': fid_cosmo['Omega_m'], 'h': fid_cosmo['h']})
    offset = mean - np.array([chain.markers[p] for p in params])
    offsets.append(offset)
    means.append(mean)
    colors.append(color_dict[cosmo])
    markers.append(marker_dict[cosmo])
offsets = np.array(offsets)
means = np.array(means)
print(f'Number of mocks: {len(offsets)}')

samples = MCSamples(
        samples=offsets,
        names=list(params),
        labels=[r'\Delta \omega_{\rm cdm}', r'\Delta \sigma_8'],
        settings={'fine_bins_2D': 128, 'smooth_scale_1D': 0.5, 'smooth_scale_2D': 0.5},
    )


# g = plots.get_subplot_plotter()
g = plots.get_single_plotter(width_inch=4.2, ratio=1/1)
# g.plot_2d(samples, 'omega_cdm', 'sigma8_m', lims=[-0.0022, 0.0037, -0.013, 0.017], colors=['grey'])
g.plot_2d(samples, 'omega_cdm', 'sigma8_m', colors=['grey'])
g.add_x_marker(0, lw=1.0)
g.add_y_marker(0, lw=1.0)

# g.triangle_plot(
#     roots=samples,
#     markers=chain.markers,
#     params=params,
# )

colors_face = [lighten_color(color, 0.5) for color in colors]

g.fig.axes[0].scatter(offsets[:, 0], offsets[:, 1], s=15, c=colors_face, edgecolors=colors, linewidths=0.5)
g.fig.axes[0].scatter(np.nan, np.nan, color='dimgrey', marker='o', label=r'$\textrm{c000}$')
g.fig.axes[0].scatter(np.nan, np.nan, color='orange', marker='o', label=r'$\textrm{c001}$')
g.fig.axes[0].scatter(np.nan, np.nan, color='dodgerblue', marker='o', label=r'$\textrm{c002}$')
g.fig.axes[0].scatter(np.nan, np.nan, color='forestgreen', marker='o', label=r'$\textrm{c003}$')
g.fig.axes[0].scatter(np.nan, np.nan, color='crimson', marker='o', label=r'$\textrm{c004}$')

l = g.fig.axes[0].legend(fontsize=13, loc='upper left', handletextpad=0.1,
                         handlelength=0, ncol=2, columnspacing=0.5)
for text, color in zip(l.get_texts(), color_dict.values()):
    text.set_color(color)
for item in l.legendHandles:
    item.set_visible(False)

plt.savefig('bestfits_scatter.png', dpi=300, bbox_inches='tight')
plt.savefig('bestfits_scatter.pdf', bbox_inches='tight')


