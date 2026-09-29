"""
Animation of the mode sweep (lyot small, z-band, lab data).

One frame per poke (9 modes × 11 pokes = 99 frames total), ~10 s at ~10 fps.
Image formatting follows the other plots:  sqrt-scaled intensity, inferno
colormap, fixed colorbar across all frames.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from hcipy import *
from flowfs import *

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
sweep_file  = 'data/flowfs_sweep_lyotsm_14.0mm_z_9modes_2025-11-30_12_05_58.pkl'
config_file = 'configs/flowfs_lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab.json'
save_name   = 'plots/presentation/sweep_animation_lyot_sm_z_lab.gif'
fps = 10 / 99

# load data
im_frames = []
fs = []
acts = []
data = pickle.load(open(sweep_file, 'rb'))

for i in range(99):
    f, actuators, surface, images = data[i]

    im_frames.append(images)
    acts.append(actuators)


# fixed colorbar  – sqrt scaling, same as plot_difference
vmax = max(im.max() for im in im_frames) ** 0.5

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig_size = 4.0   # inches (square-ish)
fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
ax.axis('off')

im_plot = ax.imshow(
    np.clip(im_frames[0].reshape(32, 32), 0, vmax)**0.5,
    cmap='inferno', vmin=0, vmax=vmax,
    origin='lower', aspect='equal', interpolation='none',
)
ax.set_title('FLOWFS Images', fontsize='large')

cbar = fig.colorbar(im_plot, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$\sqrt{I_\mathrm{norm}}$', fontsize='small')
cbar.ax.tick_params(labelsize='x-small')

# title = ax.set_title(
#     f'{zernike_modes[0]},  poke = {poke_amp[0][0]:+.3f}',
#     fontsize='small',
# )

plt.tight_layout()

# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------
total_frames = 99  # 9 modes × 11 pokes
fps          = total_frames / 10.0    # ~10 fps → ~10 s

def _update(frame):
    im_plot.set_data(np.clip(im_frames[frame].reshape(32, 32), 0, vmax)**0.5)
    return [im_plot]

anim = animation.FuncAnimation(
    fig, _update,
    frames=total_frames,
    interval=1000.0 / fps,
    blit=True,
)

anim.save(save_name, writer='pillow', fps=fps)
plt.close(fig)
print(f'Saved to {save_name}')


# figure 2
fig_size = 4.0   # inches (square-ish)
fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
ax.axis('off')

im_plot = ax.imshow(
    np.clip(im_frames[0].reshape(32, 32), 0, vmax)**0.5,
    cmap='inferno', vmin=0, vmax=vmax,
    origin='lower', aspect='equal', interpolation='none',
)
ax.set_title('FLOWFS Images', fontsize='large')

cbar = fig.colorbar(im_plot, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$\sqrt{I_\mathrm{norm}}$', fontsize='small')
cbar.ax.tick_params(labelsize='x-small')

# title = ax.set_title(
#     f'{zernike_modes[0]},  poke = {poke_amp[0][0]:+.3f}',
#     fontsize='small',
# )

plt.tight_layout()

# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------
total_frames = 99  # 9 modes × 11 pokes
fps          = 9 / 3    # ~10 fps → ~10 s

def _update(frame):
    im_plot.set_data(np.clip(im_frames[10-frame].reshape(32, 32), 0, vmax)**0.5)
    return [im_plot]

anim = animation.FuncAnimation(
    fig, _update,
    frames=11,
    interval=1000.0 / fps,
    blit=True,
)

save_name = save_name.split('.')[0] + '_tip.gif'
anim.save(save_name, writer='pillow', fps=fps)
plt.close(fig)


# ===========================================================================
# Transition animation – on-sky, lyot small z-band
# Styled after the paper figure (cell 40 in 6_analyse_on_sky_transition.ipynb)
# ===========================================================================

mpl.rcParams.update({
    "font.size": 15,
    "font.family": "sans-serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{txfonts}",
    "legend.fontsize": "x-small",
    "legend.title_fontsize": "x-small",
    "xtick.labelsize": "x-small",
    "ytick.labelsize": "small",
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "figure.labelsize": "small",
    "axes.labelsize": "small",
    "axes.titlesize": "small",
    "axes.linewidth": 0.5,
    "lines.linewidth": 1,
    "lines.markersize": 2,
    "errorbar.capsize": 2,
})


def _crop(im, size, x, y):
    return im[int(x) - size // 2:int(x) + size // 2,
              int(y) - size // 2:int(y) + size // 2]


# --- data ---
data_dir     = 'data/reduced_on_sky/2025-04-17_074500_AlphaCen_lowfs_open_closed_loop_transition'
trans_save   = 'plots/presentation/transition_animation.gif'

datas_flowfs = read_fits(os.path.join(data_dir, 'camflowfs.fits'))
datas_sci    = read_fits(os.path.join(data_dir, 'camsci.fits'))

# science camera geometry (consistent with notebook cell 40)
cx, cy     = 538, 506          # crop centre passed as (x, y) to _crop
size       = 150
inner_r    = 60  / 5.97        # pixels
outer_r    = 200 / 5.97
ccx = ccy  = size // 2

# normalisation via DM speckle spots
flux_ratio = 9.058e-04
y_idx, x_idx = np.ogrid[:datas_sci.shape[1], :datas_sci.shape[2]]
dm_spots = [(370, 540), (643, 540)]
max_int = []
for sx, sy in dm_spots:
    m = (x_idx - sx)**2 + (y_idx - sy)**2 <= 15**2
    max_int.append(np.mean(np.percentile(datas_sci[-30:, m], 99, axis=0)))
norm_val = np.mean(max_int) / flux_ratio

# annulus mask and pre-compute metric
xx, yy = np.meshgrid(np.arange(size), np.arange(size))
ann    = ((xx - ccx)**2 + (yy - ccy)**2 > inner_r**2) & \
         ((xx - ccx)**2 + (yy - ccy)**2 < outer_r**2)
c_vals = np.array([np.mean(_crop(im, size, cx, cy)[ann]) / norm_val for im in datas_sci])
d_vals = np.array([np.max( _crop(im, size, cx, cy)[ann]) / norm_val for im in datas_sci])

n = len(datas_flowfs)      # science-camera frames (== len(datas_sci))

# ---------------------------------------------------------------------------
# Animation tuning parameters
# ---------------------------------------------------------------------------
fps_display   = 20    # playback fps (FLOWFS updates every frame; science/metric update when the science frame changes)
flowfs_offset = 0     # integer raw-FLOWFS frame offset – tweak to align the transition

# --- real-time fps from raw science-camera FITS timestamps ---

dt_camsci_s = 0.22734744

fps_trans = 1.0 / dt_camsci_s
print(f'Science camera: {fps_trans:.2f} fps  ({dt_camsci_s*1e3:.1f} ms / frame)')

# --- raw high-cadence FLOWFS frames ---
_raw_flowfs_dir = os.path.join(
    'data/raw_on_sky',
    '2025-04-17_074500_AlphaCen_lowfs_open_closed_loop_transition',
    'camflowfs',
)

print(f'Looking for raw FLOWFS frames in {_raw_flowfs_dir}')
if os.path.isdir(_raw_flowfs_dir):
    from astropy.io import fits as _afits
    _ff = sorted(f for f in os.listdir(_raw_flowfs_dir) if f.endswith('.fits'))
    datas_flowfs_raw = np.concatenate(
        [_afits.getdata(os.path.join(_raw_flowfs_dir, f)) for f in _ff]
    ).astype(np.float32)
    print(f'Loaded {len(datas_flowfs_raw)} raw FLOWFS frames')
else:
    print('Raw FLOWFS dir not found – falling back to averaged frames')
    datas_flowfs_raw = datas_flowfs

dt_flowfs_raw = (n * dt_camsci_s) / len(datas_flowfs_raw)
print(f'FLOWFS raw cadence: {1/dt_flowfs_raw:.0f} fps')

# total animation duration and display frames
total_duration_s  = n * dt_camsci_s
total_disp_frames = int(total_duration_s * fps_display)

# fixed colour limits (sqrt-scaled, inferno – same as paper plot)
vmax_f = np.max(datas_flowfs_raw) ** 0.5
vmin_f = 0.0
vmax_s = np.nanmax(np.clip(_crop(datas_sci[0], size, cx, cy), 0, None) ** 0.5)
vmin_s = 0.0

# --- figure layout (paper width, 2-row: images top, metric bottom) ---
width = 523.5307 / 72
fig   = plt.figure(figsize=(width, width*3/4), dpi=150)
gs    = fig.add_gridspec(2, 2,
                         width_ratios=[1, 1],
                         height_ratios=[3, 2],
                         hspace=0.30, wspace=0.40)

ax_f = fig.add_subplot(gs[0, 0])
ax_s = fig.add_subplot(gs[0, 1])
ax_m = fig.add_subplot(gs[1, :])

# --- FLOWFS image panel ---
im_f = ax_f.imshow(datas_flowfs_raw[0] ** 0.5,
                   cmap='inferno', vmin=vmin_f, vmax=vmax_f,
                   origin='lower', aspect='equal', interpolation='none')
ax_f.set_xticks([])
ax_f.set_yticks([])
for sp in ax_f.spines.values(): sp.set_visible(False)
ax_f.set_title('FLOWFS images', fontsize='small')
cax_f = inset_axes(ax_f, width="4%", height="100%", loc="lower left",
                   bbox_to_anchor=(1.02, 0, 1, 1), bbox_transform=ax_f.transAxes,
                   borderpad=0)
fig.colorbar(im_f, cax=cax_f).set_label(r'$\sqrt{\textrm{counts}}$', fontsize='x-small')

# --- Science image panel ---
im_s = ax_s.imshow(np.clip(_crop(datas_sci[0], size, cx, cy), 0, None) ** 0.5,
                   cmap='inferno', vmin=vmin_s, vmax=vmax_s,
                   aspect='equal', interpolation='none')
patch_inner, = ax_s.plot([], [], color='r', ls='-',  lw=0.8, label=r'$3\,\lambda/D$')
patch_outer, = ax_s.plot([], [], color='r', ls='--', lw=0.8, label=r'$10\,\lambda/D$')
theta = np.linspace(0, 2 * np.pi, 256)
ax_s.plot(ccx + inner_r * np.cos(theta), ccy + inner_r * np.sin(theta),
          color='r', ls='-',  lw=0.8)
ax_s.plot(ccx + outer_r * np.cos(theta), ccy + outer_r * np.sin(theta),
          color='r', ls='--', lw=0.8)
ax_s.legend(handles=[patch_inner, patch_outer], loc='upper left',
            framealpha=0.6, fontsize='x-small')
ax_s.set_xticks([]); ax_s.set_yticks([])
for sp in ax_s.spines.values(): sp.set_visible(False)
ax_s.set_title('Coronagraphic Science Images', fontsize='small')
cax_s = inset_axes(ax_s, width="4%", height="100%", loc="lower left",
                   bbox_to_anchor=(1.02, 0, 1, 1), bbox_transform=ax_s.transAxes,
                   borderpad=0)
fig.colorbar(im_s, cax=cax_s).set_label(r'$\sqrt{\textrm{counts}}$', fontsize='x-small')

# --- Metric panel ---
t_axis = np.arange(n) * dt_camsci_s
ax_m.plot(t_axis, c_vals, 'C1-', label='Mean intensity')
ax_m.plot(t_axis, d_vals, 'C0-', label='Peak intensity')
vline = ax_m.axvline(0, color='k', ls='--', lw=0.8, zorder=5)
ax_m.set_xlim(0, (n - 1) * dt_camsci_s)
ax_m.set_ylim(np.nanmin(c_vals) * 0.8, np.nanmax(d_vals) * 1.5)
ax_m.set_yscale('log')
ax_m.set_xlabel('Time (s)')
ax_m.set_ylabel(r'$I / I_\star$')
ax_m.set_title(r'Normalised intensity in 3--10 $\lambda/D$ annulus')
ax_m.legend(loc='upper right')
ax_m.tick_params(direction='out', which='both', length=2)

plt.tight_layout()


def _update_trans(i):
    t_real   = i / fps_display
    sci_idx  = min(int(t_real / dt_camsci_s), n - 1)
    raw_idx  = int(np.clip(int(t_real / dt_flowfs_raw) + flowfs_offset,
                           0, len(datas_flowfs_raw) - 1))
    im_f.set_data(datas_flowfs_raw[raw_idx] ** 0.5)
    im_s.set_data(np.clip(_crop(datas_sci[sci_idx], size, cx, cy), 0, None) ** 0.5)
    vline.set_xdata([t_real, t_real])
    return [im_f, im_s, vline]


anim_trans = animation.FuncAnimation(
    fig, _update_trans,
    frames=total_disp_frames,
    interval=1000.0 / fps_display,
    blit=True,
)

still_save = trans_save.replace('.gif', '_frame0.png')
fig.savefig(still_save, dpi=150, transparent=True)
print(f'Saved still to {still_save}')

anim_trans.save(trans_save, writer='pillow', fps=fps_display)
plt.close(fig)
print(f'Saved to {trans_save}')
print(f'Saved to {save_name}')

# ---------------------------------------------------------------------------
# Open / closed loop comparison – first and last science frame
# ---------------------------------------------------------------------------
im_open   = np.clip(_crop(datas_sci[0],  size, cx, cy), 0, None) ** 0.5
im_closed = np.clip(_crop(datas_sci[-1], size, cx, cy), 0, None) ** 0.5
vmax_oc   = max(im_open.max(), im_closed.max())

theta = np.linspace(0, 2 * np.pi, 256)

fig_oc, axes_oc = plt.subplots(1, 2, figsize=(width * 0.55, width * 0.28), dpi=150,
                               layout='constrained')
for ax_oc, im_oc, title_oc in zip(
    axes_oc,
    [im_open, im_closed],
    ['Open loop', 'Closed loop'],
):
    mappable = ax_oc.imshow(im_oc, cmap='inferno', vmin=0, vmax=vmax_oc,
                            aspect='equal', interpolation='none')
    ax_oc.plot(ccx + inner_r * np.cos(theta), ccy + inner_r * np.sin(theta),
               color='r', ls='-',  lw=0.8)
    ax_oc.plot(ccx + outer_r * np.cos(theta), ccy + outer_r * np.sin(theta),
               color='r', ls='--', lw=0.8)
    ax_oc.set_xticks([]); ax_oc.set_yticks([])
    for sp in ax_oc.spines.values(): sp.set_visible(False)
    ax_oc.set_title(title_oc, fontsize='small')

fig_oc.colorbar(mappable, ax=axes_oc, fraction=0.03, pad=0.02).set_label(
    r'$\sqrt{\textrm{counts}}$', fontsize='x-small')
oc_save = trans_save.replace('.gif', '_open_closed.png')
fig_oc.savefig(oc_save, dpi=150, transparent=True)
plt.close(fig_oc)
print(f'Saved open/closed comparison to {oc_save}')
