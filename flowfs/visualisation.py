import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
from hcipy import *
from flowfs.FLOWFS import FLOWFS
from flowfs.bump_mask import make_magaox_bump_mask

def centroid(im):
    x = np.arange(im.shape[1])
    y = np.arange(im.shape[0])
    X, Y = np.meshgrid(x, y)
    x = np.sum(X*im)/np.sum(im)
    y = np.sum(Y*im)/np.sum(im)
    return x, y


def plot_annuli(ax, im):
    cx, cy = centroid(im.reshape(32,32))
    for r in (1, 4):
        circle = Circle((cx,cy), r*15e-6, fill=False, edgecolor='red', linewidth=1, ls='--')
        ax.add_patch(circle)

def plot_mask_size(ax, im):
    cx, cy = centroid(im.reshape(32,32))
    circle = Circle((cy,cx), 7.640467585993802, fill=False, edgecolor='red', linewidth=1, ls='--')
    ax.add_patch(circle)

    circle = Circle((cy,cx), 453/272 * 7.640467585993802, fill=False, edgecolor='red', linewidth=1, ls='--')
    ax.add_patch(circle)

def plot_difference(im_ref, model, titles=['']*3, figure_width=12):
    fig = plt.figure(figsize=(figure_width, figure_width/3))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1, 1, 0.30, 1],
        wspace=0.06,
        left=0.03, right=0.90
    )
    ax = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 3]),
    ]

    vmax = im_ref.max()**.5
    im0 = imshow_field(im_ref**.5, cmap='inferno', vmin=0, vmax=vmax, ax=ax[0])
    ax[0].set_title(titles[0])

    im1 = imshow_field(model**.5, cmap='inferno', vmin=0, vmax=vmax, ax=ax[1])
    ax[1].set_title(titles[1])

    p = 1
    vmax = np.max(np.abs(im_ref - model))/im_ref.max()
    im2 = imshow_field((im_ref - model)/im_ref.max(), cmap='bwr', vmin=-1 * p * vmax, vmax=p * vmax, ax=ax[2])
    ax[2].set_title(titles[2])

    for a in ax:
        plot_annuli(a, im_ref)
        a.axis('off')

    pos1 = ax[1].get_position()
    pos2 = ax[2].get_position()
    cb_w = 0.010
    cb_pad = 0.006

    cax_shared = fig.add_axes([pos1.x1 + cb_pad, pos1.y0, cb_w, pos1.height])
    cax_diff = fig.add_axes([pos2.x1 + cb_pad, pos2.y0, cb_w, pos2.height])

    cb1 = fig.colorbar(im1, cax=cax_shared, orientation='vertical') 
    cb1.set_label('$\sqrt{I_{\mathrm{normalised}}}$', size='medium')
    cb2 = fig.colorbar(im2, cax=cax_diff, orientation='vertical')
    cb2.set_label('Normalised Residual', size='medium')

def plot_sweep(acts_sweep, sweep_params, sweep_params_linear=None, slopes=None, biases=None, title='', save_name=None):

    n_frames = sweep_params.shape[1]
    print(n_frames)
    zernike_modes = [
        "Tip", "Tilt", "Defocus", "Oblique astigmatism", "Vertical astigmatism", "Vertical coma", "Horizontal coma", "Vertical trefoil", "Oblique trefoil"
    ]

    num_acts = len(sweep_params) // len(zernike_modes)
    
    def slope_loss(theta, acts, sweep_params):

        slope = theta[0]
        biases = theta[1:]

        loss = 0
        for i in range(2):
            fit_xs = []
            fit_ys = []
            for j in range(num_acts):
                for k in range(len(sweep_params[i*num_acts + j])):
                    fit_xs.append(acts[i*num_acts + j][i])
                    fit_ys.append((sweep_params[i*num_acts + j][k][i] ))
            fit_xs = np.array(fit_xs)
            fit_ys = np.array(fit_ys)

            fit_ys *= (np.sign(np.median(fit_ys[:len(fit_ys)//2]) - np.median(fit_ys[len(fit_ys)//2:])))


            loss += np.sum(np.square(fit_xs * slope + biases[i] - fit_ys))
            # print(loss, np.sum(np.square(fit_xs * slope + biases[i] - fit_ys)))
        return loss
    
    def linear_abs(x, a):
        return abs(a * (x))

    def linear(x, *t):
        return t[0] * x.reshape(-1, num_acts) + np.array(t[1:]).reshape(-1, 1) 
    
    def linear_fixed_slope(x, a, b):
        return np.sign(a) * slope * x + b


    if slopes is None:
        res = minimize(slope_loss, x0=np.array([10] + [0.0]*9), args=(acts_sweep, sweep_params), method='Nelder-Mead', options={'maxiter': 10000})

        slope = res.x[0]
    else:
        slope = abs(slopes[0])

    if biases is None:
        biases = []
    if slopes is None:
        slopes = []

    fig, ax = plt.subplots(3, 3, figsize=(15, 10),)

    for i in range(9):
        xs = []
        ys = []
        ys_linear = []

        for j in range(num_acts):
            for k in range(n_frames):
                xs.append(acts_sweep[i*num_acts + j][i])
                ys.append(sweep_params[i*num_acts + j][k][i])
                ys_linear.append(sweep_params_linear[i*num_acts + j][k][i])

        xs = np.array(xs)
        ys = np.array(ys)
        ys_linear = np.array(ys_linear)

        xs = np.mean(xs.reshape(-1, n_frames), axis=1)
        error = np.std(ys.reshape(-1, n_frames), axis=1) 
        ys = np.mean(ys.reshape(-1, n_frames), axis=1)

        ys_linear = np.mean(ys_linear.reshape(xs.shape[0], -1), axis=1)
        error_linear = np.std(ys_linear.reshape(xs.shape[0], -1), axis=1)

        l1 = ax[i//3, i%3].errorbar(xs * slopes[i], ys - biases[i], yerr=error, fmt='o', ms=5, capsize=4) 

        # l2 = ax[i//3, i%3].errorbar(xs * slopes[i], ys_linear - biases[i], yerr=error_linear, fmt='^', ms=5, capsize=4, c='C1')

        l3 = ax[i//3, i%3].errorbar([-3, 3], [-3, 3], c='k', ls='--', alpha=0.5)

        ax[i//3, i%3].axhline(0, color='k', ls='--', lw=1, alpha=0.5)
        
        ax[i//3, i%3].set_title(zernike_modes[i])
        ax[i//3, i%3].text(0.88, -1.55, f'Bias: {biases[i]:>6.2f}')
        xmax = np.max(xs * slopes[i]) *1.2
        ax[i//3, i%3].set_xlim([-xmax, xmax])
        ax[i//3, i%3].set_ylim([-xmax, xmax])
        # Add 0.5-spaced ticks within the current limits on both axes
        xlo, xhi = ax[i//3, i%3].get_xlim()
        ylo, yhi = ax[i//3, i%3].get_ylim()
        xticks = np.arange(np.floor(xlo * 2) / 2, np.ceil(xhi * 2) / 2 + 0.25, 0.5)
        yticks = np.arange(np.floor(ylo * 2) / 2, np.ceil(yhi * 2) / 2 + 0.25, 0.5)
        ax[i//3, i%3].set_xticks(xticks[(xticks >= xlo) & (xticks <= xhi)])
        ax[i//3, i%3].set_yticks(yticks[(yticks >= ylo) & (yticks <= yhi)])

        if i // 3 == 2 and i%3 ==1:
            ax[i//3, i%3].set_xlabel('Input RMS actuators', fontsize=14)
        if i % 3 == 0 and i // 3 ==1:
            ax[i//3, i%3].set_ylabel('Measured RMS (rad)', fontsize=14)
        
        # ax[i//3, i%3].legend()

    # add crosstalk plots
    for i in range(9):
        for j in range(9):
            if j != i:
                # print(sweep_params[i*11:(i+1)*11, j].shape, (xs * slopes[i]).shape)

                # y_ = np.mean(sweep_params[i*11:(i+1)*11, :, j], axis=1)
                y_ = np.mean(sweep_params[i*num_acts:(i+1)*num_acts, :, j], axis=1)
                y_ -= biases[j]
                l2 = ax[i//3, i%3].errorbar(xs * slopes[i], y_, fmt='o', ms=5, capsize=4, alpha=0.2, c='C2', label=f'Crosstalk {zernike_modes[j]}')
                if np.mean(np.abs(y_)) > 0.25:
                    print(f'Warning: {zernike_modes[i]} has significant crosstalk from {zernike_modes[j]}')


    ax[2,1].legend(handles = [l1, l2, l3] , labels=['Non-linear', 'Crosstalk', f'Ideal (slope={slope:.2f})'], loc='upper center',  bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncols=1)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.suptitle(title, fontsize=16)
    
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight', transparent=True)
    else:
        plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(15, 10),)

    for i in range(9):
        ys_linear = []

        for j in range(num_acts):
            for k in range(n_frames):
                ys_linear.append(sweep_params_linear[i*num_acts + j][k][i])

        xs = np.array(xs)
        ys_linear = np.array(ys_linear)
        error_linear = np.std(ys_linear.reshape(xs.shape[0], -1), axis=1)
        ys_linear = np.mean(ys_linear.reshape(xs.shape[0], -1), axis=1)

        l1 = ax[i//3, i%3].errorbar(xs * slopes[i], ys_linear, yerr=error_linear, fmt='^', ms=5, capsize=4, c='C1') # don't subtract bias as the liknear reconstructor is calculated with the bias as the 0-point

        l3 = ax[i//3, i%3].errorbar([-3, 3], [-3, 3], c='k', ls='--', alpha=0.5)
        
        ax[i//3, i%3].set_xlim([-xmax, xmax])
        ax[i//3, i%3].set_ylim([-xmax, xmax])
        ax[i//3, i%3].axhline(0, color='k', ls='--', lw=1, alpha=0.5)

        ax[i//3, i%3].text(0.88, -1.55, f'Bias: {biases[i]:>5.2f}')
        ax[i//3, i%3].set_title(zernike_modes[i])

        if i // 3 == 2 and i%3 ==1:
            ax[i//3, i%3].set_xlabel('Input RMS actuators', fontsize=14)
        if i % 3 == 0 and i // 3 ==1:
            ax[i//3, i%3].set_ylabel('Measured RMS (rad)', fontsize=14)
        
        # ax[i//3, i%3].legend()

    for i in range(9):
        for j in range(9):
            if j != i:

                y_ = np.mean(sweep_params_linear[i*num_acts:(i+1)*num_acts, :, j], axis=1)
                # y_ -= biases[j]
                l2 = ax[i//3, i%3].errorbar(xs * slopes[i], y_, fmt='o', ms=5, capsize=4, alpha=0.2, c='C2', label=f'Crosstalk {zernike_modes[j]}')
                if np.mean(np.abs(y_)) > 0.25:
                    print(f'Warning: {zernike_modes[i]} has significant crosstalk from {zernike_modes[j]}')


    ax[2,1].legend(handles = [l1, l2, l3] , labels=[ 'Linear', 'Crosstalk', f'Ideal (slope={slope:.2f})'], loc='upper center',  bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncols=1)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.suptitle(title, fontsize=16)
    
    if save_name is not None:
        plt.savefig(save_name[:-4] + "_linear" + save_name[-4:], bbox_inches='tight', transparent=True)

    return biases, slopes


def plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=None, biases=None, title='', save_name=None, plot_crosstalk=False, figure_width=15, show_linear=True, show_stats=True):
    """Plot non-linear and linear sweep results together, optionally adding crosstalk overlays."""

    n_frames = sweep_params.shape[1]
    zernike_modes = [
        "Tip", "Tilt", "Defocus", "Oblique astigmatism", "Vertical astigmatism",
        "Vertical coma", "Horizontal coma", "Vertical trefoil", "Oblique trefoil"
    ]
    num_acts = len(sweep_params) // len(zernike_modes)
    use_tex = plt.rcParams.get('text.usetex', False)

    def _xy(i, data):
        """Averaged xs, ys, and per-point std for mode i over n_frames."""
        xs = np.array([acts_sweep[i * num_acts + j][i] for j in range(num_acts)])
        raw = data[i * num_acts:(i + 1) * num_acts, :, i]  # (num_acts, n_frames)
        return xs, np.mean(raw, axis=1), np.std(raw, axis=1)

    # --- fit slopes and biases ---
    if slopes is None:
        slopes, biases = [], []
        for i in range(9):
            xs, ys, _ = _xy(i, sweep_params)
            k = 3
            X = np.column_stack((xs[k:-k], np.ones(len(xs) - 2 * k)))
            slope_i = np.linalg.lstsq(X, ys[k:-k], rcond=None)[0][0]
            slopes.append(slope_i)
            biases.append(np.median(ys - xs * slope_i))
    else:
        biases = []
        for i in range(9):
            xs, ys, _ = _xy(i, sweep_params)
            biases.append(np.median(ys - slopes[i] * xs))

    # --- figure ---
    fig, ax = plt.subplots(3, 3, figsize=(figure_width, figure_width * 3/4), sharex=False, sharey=False)
    for a in ax.flat:
        a.tick_params(direction='out', which='both', length=2 )

    first_nonlin = first_lin = first_diag = first_nonlin_ct = first_lin_ct = None
    xs_by_mode = []
    largest_xmax = 0

    for i in range(9):
        a = ax[i // 3, i % 3]
        xs, ys, err = _xy(i, sweep_params)
        _, ys_lin, err_lin = _xy(i, sweep_params_linear)
        xs_by_mode.append(xs)

        l1 = a.errorbar(xs * slopes[i], ys - biases[i], yerr=err, fmt='o', c='C0')
        if show_linear:
            l2 = a.errorbar(xs * slopes[i], ys_lin, yerr=err_lin, fmt='^', c='C1')
        l3 = a.errorbar([-3, 3], [-3, 3], c='k', ls='-.', lw=0.75, alpha=0.7, zorder=-1)
        a.axhline(0, color='k', ls='--', lw=0.75, alpha=0.5)

        if first_nonlin is None: first_nonlin = l1
        if show_linear and first_lin is None: first_lin = l2
        if first_diag   is None: first_diag   = l3

        # mode name — top left, bold
        mode_label = rf"\textbf{{{zernike_modes[i]}}}" if use_tex else zernike_modes[i]
        a.text(0.05, 0.95, mode_label, transform=a.transAxes,
               ha='left', va='top', fontsize='small')

        if show_stats:
            if use_tex:
                stats_text = (
                    r"\begin{tabular}{@{}lr@{}}"
                    rf"Bias: & {biases[i]:+.2f}\\\\"
                    rf"Slope: & {slopes[i]:+.2f}"
                    r"\end{tabular}"
                )
            else:
                stats_text = f"Bias: {biases[i]:+.2f}\nSlope: {slopes[i]:+.2f}"
            a.text(0.97, 0.03, stats_text, transform=a.transAxes,
                   ha='right', va='bottom', fontsize='x-small',
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7', alpha=0.9))

        xmax = np.max(np.abs(xs * slopes[i])) * 1.4
        ticks = np.concatenate([np.arange(0, xmax, 0.5), np.arange(0, -xmax, -0.5)])
        a.set_xticks(ticks)
        a.set_yticks(ticks)
        a.set_xlim([-xmax, xmax])
        a.set_ylim([-xmax, xmax])

        if i // 3 == 2 and i % 3 == 1:
            a.set_xlabel('Input Zernike coefficients (rad)')
        if i % 3 == 0 and i // 3 == 1:
            a.set_ylabel('Recovered Zernike coefficients (rad)')

    if plot_crosstalk:
        for i in range(9):
            for j in range(9):
                if j == i:
                    continue
                a = ax[i // 3, i % 3]
                y_nl = np.mean(sweep_params[i*num_acts:(i+1)*num_acts, :, j], axis=1) - biases[j]
                h_nl = a.errorbar(xs_by_mode[i] * slopes[i], y_nl, fmt='o', alpha=0.2, c='C2')
                y_l  = np.mean(sweep_params_linear[i*num_acts:(i+1)*num_acts, :, j], axis=1)
                h_l  = a.errorbar(xs_by_mode[i] * slopes[i], y_l,  fmt='^', alpha=0.2, c='C3')
                if first_nonlin_ct is None: first_nonlin_ct = h_nl
                if first_lin_ct    is None: first_lin_ct    = h_l

    legend_handles = [first_nonlin]
    legend_labels  = ['Non-linear']
    if show_linear:
        legend_handles.append(first_lin)
        legend_labels.append('Linear')
    legend_handles.append(first_diag)
    legend_labels.append('Ideal')
    if plot_crosstalk:
        ct_pos = 1 if not show_linear else 2
        legend_handles.insert(ct_pos, first_nonlin_ct)
        legend_labels.insert(ct_pos, 'Crosstalk (non-linear)')
        if show_linear:
            legend_handles.insert(ct_pos + 2, first_lin_ct)
            legend_labels.insert(ct_pos + 2, 'Crosstalk (linear)')

    fig.legend(handles=legend_handles, labels=legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.965),
               fancybox=True, shadow=True, ncols=len(legend_labels))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.15, hspace=0.15)

    if title:
        plt.suptitle(title, y=0.995)
    if save_name is not None:
        plt.savefig(save_name, transparent=True)

    return biases, slopes


def sliding_quantiles(x, y, window_size, num_eval_points):
    x = np.array(x)
    y = np.array(y)

    x_sorted = np.sort(x)
    y_sorted = y[np.argsort(x)]

    eval_range = np.linspace(np.min(x), np.max(x) , num_eval_points)

    window_size = eval_range[1] - eval_range[0]

    x_quantiles = []
    y_quantiles = []
    for i, eval_point in enumerate(eval_range):
        # indices = (x > eval_point - window_size/2) & (x < eval_point + window_size/2)
        indices = (x > eval_point) & (x < eval_point + window_size)
        # print(np.sum(indices))
        if np.sum(indices) > 0:
            y_window = y[indices]
            x_quantiles.append(eval_point + window_size / 2)
            y_quantiles.append(np.percentile(y_window, [2.5, 16, 50, 84, 97.5]))
            # m = np.mean(y_window)
            # s = np.std(y_window)
            # y_quantiles.append( [m - 2*s, m - s, m, m + s, m + 2*s] )
    
    return np.array(x_quantiles), np.array(y_quantiles)
    
def plot_sliding_quantiles(x, y, window_size, num_eval_points, label, color, ls='-'):
    x_quantiles, y_quantiles = sliding_quantiles(x, y, window_size, num_eval_points)
    plt.plot(x_quantiles, y_quantiles[:, 2], label=f'{label} (min {np.min(y_quantiles[:,2]):.1f}nm)', color=color, ls=ls)
    # plt.fill_between(x_quantiles, y_quantiles[:, 0], y_quantiles[:, 4], color=color, alpha=0.2)
    plt.fill_between(x_quantiles, y_quantiles[:, 1], y_quantiles[:, 3], color=color, alpha=0.2)
    # if plot_convergence:
    #     plt.axhline(y_quantiles[0, 2], color=color, ls='--', alpha=0.5, label='Convergence {} ({:.1f} nm)'.format(label, y_quantiles[0, 2]))

def injection_reconstruction_plot(acts, fitted_params, fitted_params_linear, slopes, biases, wavelength_wfs=908e-9, save_name=None, title='', xlim=None, ylim=None, num_bins=10):
    # rms = lambda x: np.sqrt(np.mean(x**2))
    rms = lambda x: np.std(x) # this is stupid
    def dm2zernike(actuators, slopes):
        return slopes * actuators #+ biases

    flowfs = FLOWFS()
    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)
    
    xs = []
    ys = []

    for i in range(len(fitted_params)):
        input_modes = dm2zernike(acts[i][:9], slopes) + biases[:9]
        # rms_ = rms(np.mean(fitted_params[i,:,:9], axis=0) - input_modes[:9]) 
        rms_ = flowfs.rms(np.mean(fitted_params[i,:,:9], axis=0) - input_modes[:9], magellan_aperture)
        # xs.append(rms(input_modes))
        xs.append(flowfs.rms(input_modes, magellan_aperture))
        ys.append(rms_)

    xs = np.array(xs)
    ys = np.array(ys)

    plot_sliding_quantiles(wavelength_wfs*1e9/(2*np.pi)*xs, wavelength_wfs*1e9/(2*np.pi)*ys, 1, num_bins, 'Non-linear', 'C0')

    xs = []
    ys = []

    for i in range(len(fitted_params)):
        input_modes = dm2zernike(acts[i][:9], slopes) #+ biases[:9]
        rms_ = flowfs.rms(np.mean(fitted_params_linear[i,:,:9], axis=0) - input_modes[:9], magellan_aperture)

        xs.append(flowfs.rms(input_modes + biases[:9], magellan_aperture))
        ys.append(rms_)

    xs = np.array(xs)
    ys = np.array(ys)

    plot_sliding_quantiles(wavelength_wfs*1e9/(2*np.pi)*xs, wavelength_wfs*1e9/(2*np.pi)*ys, 1, num_bins, 'Linear', 'C1')

    xmax = np.max(wavelength_wfs*1e9/(2*np.pi)*xs)
    xlim = xlim if xlim is not None else xmax
    ylim = ylim if ylim is not None else xmax
    
    plt.plot([0,xlim], [0,ylim], 'k-.', alpha=0.5, linewidth=.75)
    legend = plt.legend(title=f'{title}')
    # left-align legend entries and title for consistent layout
    legend._legend_box.align = "left"
    legend.get_title().set_multialignment("left")
    # plt.title(f'{title}')
    plt.xlabel('input rms (nm)')
    plt.ylabel('residual rms (nm)')
    plt.xlim([0, xlim])
    plt.ylim([0, ylim])
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight', transparent=True)