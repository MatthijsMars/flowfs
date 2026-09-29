#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from flowfs import *
import pickle
import json
import os
from hcipy import *
#%%
def sweep_plot():
    mpl.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{txfonts}",
    "legend.fontsize": "small",
    "legend.title_fontsize": "x-small",
    "xtick.labelsize": "x-small",
    "ytick.labelsize": "x-small",
    "figure.labelsize": "small",
    "axes.labelsize": "small",
    "axes.titlesize": "small",
    "axes.linewidth": 0.5,
    "lines.linewidth": 1,
    "lines.markersize": 2,
    "errorbar.capsize": 2,
    })

    total_width = 523.5307 / 72  # set final figure width in inches

    sweep_params_linear = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params_linear.npy")
    sweep_params = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params.npy")
    acts_sweep = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/acts.npy")

    biases, slopes = plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=None, biases=None, save_name='plots/paper/lyot_sm_sweep_lab_combined.svg', plot_crosstalk=True, figure_width=total_width)
    plt.close()
    np.save("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy", slopes)
    np.save("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/biases.npy", biases)
    # biases, slopes = plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=None, biases=None, title='lab: Lyot small (z-band)', save_name='plots/poster/lyot_sm_sweep_lab_combined.png', plot_crosstalk=True)

    sweep_params_linear = np.load("results/lyot_sm_14mm_z_9modes_2025-04-15_02_21_15_on_sky/sweep_params_linear.npy")
    sweep_params = np.load("results/lyot_sm_14mm_z_9modes_2025-04-15_02_21_15_on_sky/sweep_params.npy")
    acts_sweep = np.load("results/lyot_sm_14mm_z_9modes_2025-04-15_02_21_15_on_sky/acts.npy")

    
    plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=slopes, biases=None, save_name='plots/poster/lyot_sm_sweep_on_sky_combined.svg', plot_crosstalk=True, figure_width=total_width)
    # biases, slopes = plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=slopes, biases=None, title='On-sky: lyot small (z-band)', save_name='plots/poster/lyot_sm_sweep_on_sky_combined.png', plot_crosstalk=True)
    plt.close(plt.gcf())

    sweep_params_linear = np.load("results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/sweep_params_linear.npy")
    sweep_params = np.load("results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/sweep_params.npy")
    acts_sweep = np.load("results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/acts.npy")

    biases, slopes = plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=None, biases=None, title='lab: Lyot large (z-band)', save_name='plots/paper/lyot_lg_sweep_lab_combined.svg', plot_crosstalk=True, figure_width=total_width)
    # biases, slopes = plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=None, biases=None, title='lab: Lyot large (z-band)', save_name='plots/paper/lyot_lg_sweep_lab_combined.png', plot_crosstalk=True)
    plt.close(plt.gcf())
    np.save("results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/slopes.npy", slopes)
    np.save("results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/biases.npy", biases)

    sweep_params_linear = np.load("results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/sweep_params_linear.npy")
    sweep_params = np.load("results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/sweep_params.npy")
    acts_sweep = np.load("results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/acts.npy")

    biases, slopes = plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=slopes, biases=None, save_name='plots/paper/lyot_lg_sweep_on_sky_combined.svg', plot_crosstalk=True, figure_width=total_width)
    # biases, slopes = plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=slopes, biases=None, title='On-sky: lyot large (z-band)', save_name='plots/paper/lyot_lg_sweep_on_sky_combined.png', plot_crosstalk=True)
    plt.close(plt.gcf())
    np.save("results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/slopes.npy", slopes)
    np.save("results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/biases.npy", biases)

def injection_retrieval_plots():
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",
        "axes.labelsize": "small",
        "axes.titlesize": "small",
        "font.size": 29,          # baseline, mapped to LaTeX
        "legend.fontsize": "xx-small",
        "legend.title_fontsize": "x-small",
        "xtick.labelsize": "x-small",
        "ytick.labelsize": "x-small",
        "figure.labelsize": "small",
    })

    def _load_case(paths):
        return (
            np.load(paths["acts"]),
            np.load(paths["fitted"]),
            np.load(paths["fitted_linear"]),
            np.load(paths["slopes"]),
            np.load(paths["biases"]),
        )

    total_width = 940 / 72  # set final figure width in inches
    panel_ratio = 3 / 4
    pad = 0.6
    panel_width = (total_width - pad) / 2
    panel_height = panel_width * panel_ratio
    fig_width = total_width
    fig_height = panel_height * 2 + pad

    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    lim = 201
    num_bins = 10

    cases = [
        {
            "case": "lyot_sm_lab_z",
            "title": "Lab: lyot small (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_acts.npy",
            "fitted": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy",
            "biases": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/biases.npy",
            "background": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/background.npy",
        },
        {
            "case": "lyot_sm_lab_r",
            "title": "Lab: lyot small (r-band)",
            "wavelength": 615e-9,
            "acts": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_acts.npy",
            "fitted": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/slopes.npy",
            "biases": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/biases.npy",
            "background": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/background.npy",
        },
        {
            "case": "lyot_lg_lab_z",
            "title": "Lab: lyot large (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_acts.npy",
            "fitted": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/slopes.npy",
            "biases": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/biases.npy",
            "background": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/background.npy",
        },
        {
            "case": "lyot_lg_on_sky_z",
            "title": "On-sky: lyot large (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_acts.npy",
            "fitted": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/slopes.npy",
            "biases": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/biases.npy",
            "background": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/background.npy",
        },
    ]

    for idx, case in enumerate(cases):
        acts, fitted, fitted_lin, slopes, biases = _load_case(case)
        fig, ax = plt.subplots(figsize=(8, 6))
        slopes = np.load(case["slopes"])
        biases = np.load(case["biases"])
        fitted = np.load(case["fitted"])
        fitted_lin = np.load(case["fitted_linear"])

        plt.sca(ax)
        injection_reconstruction_plot(
            acts[:, 0],
            fitted,
            fitted_lin,
            slopes=slopes,
            biases=biases,
            wavelength_wfs=case["wavelength"],
            save_name=None,
            title=case["title"],
            xlim=lim,
            ylim=lim,
            num_bins=num_bins,
        )
        bg = np.load(case["background"])
        plt.axhline(bg, color='k', ls='--', alpha=0.5, label='Background noise ({:.1f} nm)'.format(bg))
        plt.legend(loc='upper left', fontsize='xx-small', frameon=True)
        ax.set_title(case["title"], fontsize="small")
        ax.set_xlabel("Input RMS (nm)")
        ax.set_ylabel("Residual RMS (nm)")
        
        plt.tight_layout()
        plt.savefig(f'plots/paper/injection_{case["case"]}.svg', bbox_inches='tight')
        # plt.show()
        plt.close(fig)


#%%
def poster_injection_retrieval_plots():
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",
        "axes.labelsize": "small",
        "axes.titlesize": "small",
        "font.size": 29,          # baseline, mapped to LaTeX
        "legend.fontsize": "xx-small",
        "legend.title_fontsize": "x-small",
        "xtick.labelsize": "x-small",
        "ytick.labelsize": "x-small",
        "figure.labelsize": "small",
    })

    def _load_case(paths):
        return (
            np.load(paths["acts"]),
            np.load(paths["fitted"]),
            np.load(paths["fitted_linear"]),
            np.load(paths["slopes"]),
            np.load(paths["biases"]),
        )

    total_width = 940 / 72  # set final figure width in inches
    panel_ratio = 3 / 4
    pad = 0.6
    panel_width = (total_width - pad) / 2
    panel_height = panel_width * panel_ratio
    fig_width = total_width
    fig_height = panel_height * 2 + pad

    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    lim = 201
    num_bins = 10

    cases = [
        {
            "title": "Lab: lyot small (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_acts.npy",
            "fitted": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy",
            "biases": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/biases.npy",
            "background": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/background.npy",
        },
        {
            "title": "Lab: lyot small (r-band)",
            "wavelength": 615e-9,
            "acts": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_acts.npy",
            "fitted": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/slopes.npy",
            "biases": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/biases.npy",
            "background": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/background.npy",
        },
        {
            "title": "Lab: lyot large (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_acts.npy",
            "fitted": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/slopes.npy",
            "biases": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/biases.npy",
            "background": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/background.npy",
        },
        {
            "title": "On-sky: lyot large (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_acts.npy",
            "fitted": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/slopes.npy",
            "biases": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/biases.npy",
            "background": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/background.npy",
        },
    ]

    for idx, (ax, case) in enumerate(zip(axes.flat, cases)):
        acts, fitted, fitted_lin, slopes, biases = _load_case(case)
        plt.sca(ax)
        injection_reconstruction_plot(
            acts[:, 0],
            fitted,
            fitted_lin,
            slopes=slopes,
            biases=biases,
            wavelength_wfs=case["wavelength"],
            save_name=None,
            title=case["title"],
            xlim=lim,
            ylim=lim,
            num_bins=num_bins,
        )
        bg = np.load(case["background"])
        plt.axhline(bg, color='k', ls='--', alpha=0.5, label='Background noise ({:.1f} nm)'.format(bg))
        plt.legend(loc='upper left', fontsize='xx-small', frameon=True)
        ax.set_title(case["title"], fontsize="small")

        row, col = divmod(idx, 2)
        if row == 0:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if col == 1:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        # Strip per-axis labels; we'll add shared labels for the grid below.
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.supxlabel("Input RMS (nm)")
    fig.supylabel("Residual RMS (nm)")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08, wspace=0.05, hspace=0.05)

    # Keep figure canvas size fixed; avoid tight bbox trimming that clips labels
    # plt.savefig('plots/poster/lyot_injection_grid.pgf', transparent=True)
    plt.savefig('plots/poster/lyot_injection_grid.png')
    plt.show()

# poster_injection_retrieval_plots()


def presentation_injection_retrieval_plots():
    """Create presentation-style injection retrieval figures (paper layout, poster style)."""
    plt.style.use("configs/poster.mplstyle")

    def _load_case(paths):
        return (
            np.load(paths["acts"]),
            np.load(paths["fitted"]),
            np.load(paths["fitted_linear"]),
            np.load(paths["slopes"]),
            np.load(paths["biases"]),
        )

    cases = [
        {
            "title": "Lab: lyot small (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_acts.npy",
            "fitted": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy",
            "biases": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/biases.npy",
            "background": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/background.npy",
        },
        {
            "title": "Lab: lyot small (r-band)",
            "wavelength": 615e-9,
            "acts": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_acts.npy",
            "fitted": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/slopes.npy",
            "biases": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/biases.npy",
            "background": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/background.npy",
        },
        {
            "title": "Lab: lyot large (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_acts.npy",
            "fitted": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/slopes.npy",
            "biases": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/biases.npy",
            "background": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/background.npy",
        },
        {
            "title": "On-sky: lyot large (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_acts.npy",
            "fitted": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/slopes.npy",
            "biases": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/biases.npy",
            "background": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/background.npy",
        },
    ]

    lab_cases = cases[:3]
    on_sky_case = cases[3]
    lim = 190
    num_bins = 10

    total_width = 1318 / 72  # Poster width
    lab_height = (total_width / 3) * (3 / 4) + 1.2
    fig_lab, axes_lab = plt.subplots(1, 3, figsize=(total_width, lab_height), sharex=True, sharey=True)

    for idx, (ax, case) in enumerate(zip(axes_lab, lab_cases)):
        acts, fitted, fitted_lin, slopes, biases = _load_case(case)
        plt.sca(ax)
        injection_reconstruction_plot(
            acts[:, 0],
            fitted,
            fitted_lin,
            slopes=slopes,
            biases=biases,
            wavelength_wfs=case["wavelength"],
            save_name=None,
            title='',
            xlim=lim,
            ylim=lim,
            num_bins=num_bins,
        )
        bg = np.load(case["background"])
        ax.axhline(bg, color='k', ls='--', alpha=0.8, linewidth=1.5, label='Turbulence ({:.1f} nm)'.format(bg))
        ax.legend(loc='upper left', frameon=True)
        ax.set_title(case["title"], fontsize='small')
        ax.set_xlabel('')
        ax.set_ylabel('')
        if idx > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

    fig_lab.supxlabel(r'Input aberration $\sigma$ (nm)')
    fig_lab.supylabel(r'Residual aberration $\sigma$ (nm)')
    fig_lab.subplots_adjust(left=0.08, right=0.995, top=0.9, bottom=0.18, wspace=0.08)
    fig_lab.savefig('plots/presentation/lyot_injection_lab_row.svg', transparent=True)
    plt.close(fig_lab)

    on_sky_width = 800 / 72 # Appropriate width for presentation
    on_sky_height = on_sky_width * (3 / 4) + 1.0
    fig_on_sky, ax_on_sky = plt.subplots(1, 1, figsize=(on_sky_width, on_sky_height))

    acts, fitted, fitted_lin, slopes, biases = _load_case(on_sky_case)
    plt.sca(ax_on_sky)
    injection_reconstruction_plot(
        acts[:, 0],
        fitted,
        fitted_lin,
        slopes=slopes,
        biases=biases,
        wavelength_wfs=on_sky_case["wavelength"],
        save_name=None,
        title='',
        xlim=lim,
        ylim=lim,
        num_bins=num_bins,
    )
    bg = np.load(on_sky_case["background"])
    ax_on_sky.axhline(bg, color='k', ls='--', linewidth=1.5, alpha=0.7, label='Turbulence ({:.1f} nm)'.format(bg), zorder=0)
    ax_on_sky.legend(loc='upper left', frameon=True)
    ax_on_sky.set_title(on_sky_case["title"], fontsize='small')
    ax_on_sky.set_xlabel(r'Input aberration $\sigma$ (nm)')
    ax_on_sky.set_ylabel(r'Residual aberration $\sigma$ (nm)')
    fig_on_sky.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.2)
    fig_on_sky.savefig('plots/presentation/lyot_injection_on_sky_single.svg', transparent=True)
    plt.close(fig_on_sky)


def presentation_injection_retrieval_split():
    """Create three individual injection retrieval plots for presentation (Lab cases)."""
    plt.style.use("configs/poster.mplstyle")

    def _load_case(paths):
        return (
            np.load(paths["acts"]),
            np.load(paths["fitted"]),
            np.load(paths["fitted_linear"]),
            np.load(paths["slopes"]),
            np.load(paths["biases"]),
        )

    cases = [
        {
            "case": "lab_z_sm",
            "title": "Lab: lyot small (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_acts.npy",
            "fitted": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy",
            "biases": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/biases.npy",
            "background": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/background.npy",
        },
        {
            "case": "lab_r_sm",
            "title": "Lab: lyot small (r-band)",
            "wavelength": 615e-9,
            "acts": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_acts.npy",
            "fitted": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/slopes.npy",
            "biases": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/biases.npy",
            "background": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/background.npy",
        },
        {
            "case": "lab_z_lg",
            "title": "Lab: lyot large (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_acts.npy",
            "fitted": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/slopes.npy",
            "biases": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/biases.npy",
            "background": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/background.npy",
        },
    ]

    lim = 190
    num_bins = 10
    total_width = 1318 / 72
    lab_height = (total_width / 3) * (3 / 4) + 1.2

    # Calculate consistent axes width based on the "clean" panels (idx > 0)
    # The clean panels use left=0.05 and right=0.98
    axes_width_inches = (total_width / 3) * (0.98 - 0.05)
    
    for idx, case in enumerate(cases):
        # Adjust figure width for the first panel to accommodate the y-label
        # while keeping the actual internal axes area identical in size.
        if idx == 0:
            fig_width = axes_width_inches / (0.98 - 0.17)
        else:
            fig_width = total_width / 3
            
        fig, ax = plt.subplots(figsize=(fig_width, lab_height))
        acts, fitted, fitted_lin, slopes, biases = _load_case(case)
        plt.sca(ax)
        injection_reconstruction_plot(
            acts[:, 0],
            fitted,
            fitted_lin,
            slopes=slopes,
            biases=biases,
            wavelength_wfs=case["wavelength"],
            save_name=None,
            title='',
            xlim=lim,
            ylim=lim,
            num_bins=num_bins,
        )
        bg = np.load(case["background"])
        ax.axhline(bg, color='k', ls='--', alpha=0.8, linewidth=1.5, label='Turbulence ({:.1f} nm)'.format(bg))
        ax.legend(loc='upper left', frameon=True)
        ax.set_title(case["title"], fontsize='small')
        ax.set_xlabel(r'Input aberration $\sigma$ (nm)')
        
        if idx == 0:
            ax.set_ylabel(r'Residual aberration $\sigma$ (nm)')
            fig.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.2)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
            fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.2)
            
        fig.savefig(f'plots/presentation/lyot_injection_split_{case["case"]}.svg', transparent=True)
        plt.close(fig)


def paper_injection_retrieval_plots():
    """Create paper-style injection retrieval figures for lab (row) and on-sky (single)."""
    mpl.rcParams.update({
        "font.size": 10,
        "font.family": "sans-serif",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{txfonts}",
        "legend.fontsize": "x-small",
        "legend.title_fontsize": "x-small",
        "xtick.labelsize": "x-small",
        "ytick.labelsize": "x-small",
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

    def _load_case(paths):
        return (
            np.load(paths["acts"]),
            np.load(paths["fitted"]),
            np.load(paths["fitted_linear"]),
            np.load(paths["slopes"]),
            np.load(paths["biases"]),
        )

    cases = [
        {
            "title": "Lab: lyot small (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_acts.npy",
            "fitted": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_07_21_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy",
            "biases": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/biases.npy",
            "background": "results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/background.npy",
        },
        {
            "title": "Lab: lyot small (r-band)",
            "wavelength": 615e-9,
            "acts": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_acts.npy",
            "fitted": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_11_14_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/slopes.npy",
            "biases": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/biases.npy",
            "background": "results/lyot_sm_14mm_r_9modes_2025-11-30_12_10_46_lab/background.npy",
        },
        {
            "title": "Lab: lyot large (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_acts.npy",
            "fitted": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_42_lab/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/slopes.npy",
            "biases": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/biases.npy",
            "background": "results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/background.npy",
        },
        {
            "title": "On-sky: lyot large (z-band)",
            "wavelength": 908e-9,
            "acts": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_acts.npy",
            "fitted": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_reconstruction.npy",
            "fitted_linear": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_33_on_sky/injection_reconstruction_linear.npy",
            "slopes": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/slopes.npy",
            "biases": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/biases.npy",
            "background": "results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/background.npy",
        },
    ]

    lab_cases = cases[:3]
    on_sky_case = cases[3]
    lim = 190
    num_bins = 10

    paper_width = 523.5307 / 72
    lab_height = (paper_width / 3) * (3 / 4) + 0.55
    fig_lab, axes_lab = plt.subplots(1, 3, figsize=(paper_width, lab_height), sharex=True, sharey=True)

    for idx, (ax, case) in enumerate(zip(axes_lab, lab_cases)):
        acts, fitted, fitted_lin, slopes, biases = _load_case(case)
        plt.sca(ax)
        injection_reconstruction_plot(
            acts[:, 0],
            fitted,
            fitted_lin,
            slopes=slopes,
            biases=biases,
            wavelength_wfs=case["wavelength"],
            save_name=None,
            title='',
            xlim=lim,
            ylim=lim,
            num_bins=num_bins,
        )
        bg = np.load(case["background"])
        ax.axhline(bg, color='k', ls='--', alpha=0.8, linewidth=.75, label='Turbulence ({:.1f} nm)'.format(bg))
        ax.legend(loc='upper left', frameon=True)
        ax.set_title(case["title"], fontsize='small')
        ax.set_xlabel('')
        ax.set_ylabel('')
        if idx > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

    fig_lab.supxlabel(r'Input aberration $\sigma$ (nm)')
    fig_lab.supylabel(r'Residual aberration $\sigma$ (nm)')
    fig_lab.subplots_adjust(left=0.08, right=0.995, top=0.9, bottom=0.14, wspace=0.08)
    fig_lab.savefig('plots/paper/lyot_injection_lab_row.svg', transparent=True)
    # plt.show()
    plt.close(fig_lab)

    on_sky_width = 256.0748 / 72
    on_sky_height = on_sky_width * (3 / 4) + 0.5
    fig_on_sky, ax_on_sky = plt.subplots(1, 1, figsize=(on_sky_width, on_sky_height))

    acts, fitted, fitted_lin, slopes, biases = _load_case(on_sky_case)
    plt.sca(ax_on_sky)
    injection_reconstruction_plot(
        acts[:, 0],
        fitted,
        fitted_lin,
        slopes=slopes,
        biases=biases,
        wavelength_wfs=on_sky_case["wavelength"],
        save_name=None,
        title='',
        xlim=lim,
        ylim=lim,
        num_bins=num_bins,
    )
    bg = np.load(on_sky_case["background"])
    ax_on_sky.axhline(bg, color='k', ls='--', linewidth=.75, alpha=0.7, label='Turbulence ({:.1f} nm)'.format(bg), zorder=0)
    ax_on_sky.legend(loc='upper left', frameon=True)
    ax_on_sky.set_title(on_sky_case["title"], fontsize='small')
    ax_on_sky.set_xlabel(r'Input aberration $\sigma$ (nm)')
    ax_on_sky.set_ylabel(r'Residual aberration $\sigma$ (nm)')
    fig_on_sky.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.2)
    fig_on_sky.savefig('plots/paper/lyot_injection_on_sky_single.svg', transparent=True)
    # plt.show()
    plt.close(fig_on_sky)

#%%
def poster_sweep_plot():
    # Apply poster styling
    plt.style.use("configs/poster.mplstyle")

    sweep_params_linear = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params_linear.npy")
    sweep_params = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params.npy")
    acts_sweep = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/acts.npy")

    # Generate full grid once (no save), then trim to the top row and re-save.
    # Suppress plt.show inside plot_sweep_combined so the figure persists.
    _orig_show = plt.show
    plt.show = lambda *_, **__: None
    plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=None, biases=None, title='lab: Lyot small (z-band)', save_name=None, plot_crosstalk=True)
    plt.show = _orig_show

    fig = plt.gcf()
    axes = fig.get_axes()

    # Keep only the first three axes (top row); delete the rest
    keep = set(axes[:3])
    for ax in axes:
        if ax not in keep:
            fig.delaxes(ax)

    remaining_axes = fig.get_axes()
    if remaining_axes:
        remaining_axes[0].set_ylabel('Measured RMS (rad)',)
    if len(remaining_axes) > 1:
        remaining_axes[1].set_xlabel('Input RMS actuators')

    # Remove title from the original figure if present
    if getattr(fig, "_suptitle", None):
        fig._suptitle.remove()

    # Preserve the original axis aspect ratio measured from the generated figure,
    # but apply the requested total width of 1318/72 inches.
    fig_width_new = 1318 / 72
    if remaining_axes:
        pos = remaining_axes[0].get_position()
        fig_width_curr, fig_height_curr = fig.get_size_inches()
        ratio_curr = (pos.width * fig_width_curr) / (pos.height * fig_height_curr)
        fig_height_new = (pos.width * fig_width_new) / (pos.height * ratio_curr)
        fig.set_size_inches(fig_width_new, fig_height_new)

        if remaining_axes:
            # Recreate legend manually to include crosstalk and preserve errorbars
            sample_ax = remaining_axes[0]
            m1 = sample_ax.errorbar([], [], yerr=[], fmt='o', ms=5, capsize=4, c='C0', label='Non-linear')[0]
            m2 = sample_ax.errorbar([], [], yerr=[], fmt='^', ms=5, capsize=4, c='C1', label='Linear')[0]
            m3, = sample_ax.plot([], [], 'k--', alpha=0.5, label='Ideal')
            m4 = sample_ax.errorbar([], [], yerr=[], fmt='o', ms=5, capsize=4, c='C2', alpha=0.2, label='Crosstalk (non-linear)')[0]
            m5 = sample_ax.errorbar([], [], yerr=[], fmt='^', ms=5, capsize=4, c='C3', alpha=0.2, label='Crosstalk (linear)')[0]
            fig.legend(
                [m1, m2, m3, m4, m5],
                ['Non-linear', 'Linear', 'Ideal', 'Crosstalk (non-linear)', 'Crosstalk (linear)'],
                loc='lower center',
                ncol=5,
                frameon=True,
                bbox_to_anchor=(0.55, 0.55),
            )

    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.08, right=0.98, hspace=0.05, wspace=0.15)
    
    fig.savefig('plots/poster/lyot_sm_sweep_lab_toprow.pgf', bbox_inches='tight', transparent=True, pad_inches=0.0)
    # fig.savefig('plots/poster/lyot_sm_sweep_lab_toprow.png', bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)


def presentation_reconstruction_plot():
    """Create presentation-style reconstruction plots (paper layout, poster style)."""
    plt.style.use("configs/poster.mplstyle")
    total_width = 1318 / 72

    data = pickle.load(open("data/flowfs_sweep_lyotsm_14.0mm_z_9modes_2025-11-30_12_05_58.pkl", "rb"))
    flowfs = FLOWFS()
    flowfs.load_config("configs/flowfs_lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab.json")
    flowfs.projection(1,1)
    im_WFS = data[5][3]

    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)
    wavefront = Wavefront(magellan_aperture, flowfs.wavelength_wfs)
    wavefront.total_power = 1

    res = phase_retrieval(flowfs, flowfs.start_theta, wavefront, im_WFS, plot=False, fit_downstream=True)

    flowfs.fitted_parameters[:] = True
    flowfs._set_aberration(res.x)
    pred = flowfs.forward(wavefront)
    plot_difference(im_WFS, pred, ['Lab observation', 'Model', 'Residual'], figure_width=total_width)
    plt.savefig('plots/presentation/lyot_sm_reconstruction.svg', transparent=True)
    plt.close()


def presentation_lab_transition():
    """Create presentation-style lab transition plots (paper layout, poster style)."""
    plt.style.use("configs/poster.mplstyle")
    total_width = 1318 / 72

    data = pickle.load(open('./data/flowfs_injection_modes_slow_control_14mm_2025-06-18_10_55_15.pkl', 'rb'))
    slopes = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy")
    
    flowfs = FLOWFS()
    flowfs.load_config("configs/flowfs_lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab.json")

    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)

    fig, ax = plt.subplots(3,3, figsize = (total_width, total_width*.5), sharey=True, sharex=True)
    for j in range(len(data[:4])): 
        _, acts, ims, sci_ims = data[j]
        acts = np.array(acts) * slopes

        zernike_modes = [
            "Tip", "Tilt", "Defocus", "Oblique astigmatism", "Vertical astigmatism", "Vertical coma", "Horizontal coma", "Vertical trefoil", "Oblique trefoil"
        ]

        ax[2, 1].set_xlabel('Step')
        ax[1, 0].set_ylabel('Residual error (rad)')
        lim = np.max(np.abs(np.array(acts)[2:,] - np.median(np.array(acts)[-25:], axis=0))) *1.05
        for i in range(9):
            ax[i//3, i%3].axhline(0, ls='--', c='k', alpha=0.3, lw=1.5)
            ax[i//3, i%3].plot(np.array(acts)[2:,i] - np.median(np.array(acts)[-25:,i]), linewidth=3)
            ax[i//3, i%3].set_ylim(-lim, lim)
            ax[i//3, i%3].set_xlim(0, 109)
            ax[i//3, i%3].legend([rf'$\textbf{{{zernike_modes[i]}}}$'], handlelength=0, handletextpad=0, frameon=False, fontsize='small')
            
    plt.subplots_adjust(wspace=0.1, hspace=.1, left=0.08, right=0.98, top=0.98, bottom=0.12)
    plt.savefig('plots/presentation/lab_transition.svg', transparent=True)
    plt.close()


def presentation_sweep_plot():
    """Create presentation-style sweep plots (paper layout, poster style)."""
    plt.style.use("configs/poster.mplstyle")
    total_width = 1318 / 72

    # Lab: lyot small
    sweep_params_linear = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params_linear.npy")
    sweep_params = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params.npy")
    acts_sweep = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/acts.npy")
    plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=None, biases=None, save_name='plots/presentation/lyot_sm_sweep_lab_combined.svg', plot_crosstalk=True, figure_width=total_width)
    plt.close()

    # On-sky: lyot small
    slopes = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy")
    sweep_params_linear = np.load("results/lyot_sm_14mm_z_9modes_2025-04-15_02_21_15_on_sky/sweep_params_linear.npy")
    sweep_params = np.load("results/lyot_sm_14mm_z_9modes_2025-04-15_02_21_15_on_sky/sweep_params.npy")
    acts_sweep = np.load("results/lyot_sm_14mm_z_9modes_2025-04-15_02_21_15_on_sky/acts.npy")
    plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=slopes, biases=None, save_name='plots/presentation/lyot_sm_sweep_on_sky_combined.svg', plot_crosstalk=True, figure_width=total_width)
    plt.close()

    # Lab: lyot large
    sweep_params_linear = np.load("results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/sweep_params_linear.npy")
    sweep_params = np.load("results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/sweep_params.npy")
    acts_sweep = np.load("results/lyot_lg_14mm_z_9modes_2025-11-30_12_34_13_lab/acts.npy")
    plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=None, biases=None, title='lab: Lyot large (z-band)', save_name='plots/presentation/lyot_lg_sweep_lab_combined.svg', plot_crosstalk=True, figure_width=total_width)
    plt.close()

    # On-sky: lyot large
    sweep_params_linear = np.load("results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/sweep_params_linear.npy")
    sweep_params = np.load("results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/sweep_params.npy")
    acts_sweep = np.load("results/lyot_lg_14mm_z_9modes_2025-12-02_00_16_06_on_sky/acts.npy")
    plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear, slopes=slopes, biases=None, save_name='plots/presentation/lyot_lg_sweep_on_sky_combined.svg', plot_crosstalk=True, figure_width=total_width)
    plt.close()


def _load_lyot_sm_sweep():
    """Return (acts_sweep, sweep_params, sweep_params_linear, slopes) for lyot-sm lab."""
    sweep_params_linear = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params_linear.npy")
    sweep_params        = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params.npy")
    acts_sweep          = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/acts.npy")
    slopes              = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy")
    return acts_sweep, sweep_params, sweep_params_linear, slopes


def presentation_sweep_plot_nonlinear_only():
    """All 9 modes – non-linear points only, no crosstalk, no bias/slope box."""
    plt.style.use("configs/poster.mplstyle")
    total_width = 1318 / 72
    acts_sweep, sweep_params, sweep_params_linear, slopes = _load_lyot_sm_sweep()
    plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear,
                        slopes=slopes, biases=None,
                        save_name='plots/presentation/lyot_sm_sweep_lab_nonlinear_only.svg',
                        plot_crosstalk=False, show_linear=False, show_stats=False,
                        figure_width=total_width)
    plt.close()


def presentation_sweep_plot_both_clean():
    """All 9 modes – non-linear + linear, no crosstalk, no bias/slope box."""
    plt.style.use("configs/poster.mplstyle")
    total_width = 1318 / 72
    acts_sweep, sweep_params, sweep_params_linear, slopes = _load_lyot_sm_sweep()
    plot_sweep_combined(acts_sweep, sweep_params, sweep_params_linear,
                        slopes=slopes, biases=None,
                        save_name='plots/presentation/lyot_sm_sweep_lab_both_clean.svg',
                        plot_crosstalk=False, show_linear=True, show_stats=False,
                        figure_width=total_width)
    plt.close()


def presentation_single_tip_sweep_clean():
    """Tip mode only – non-linear points and ideal line, no linear, no crosstalk, no stats."""
    plt.style.use("configs/poster.mplstyle")

    acts_sweep          = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/acts.npy")
    sweep_params        = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params.npy")
    slopes              = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy")
    biases_saved        = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/biases.npy")

    zernike_modes = ["Tip", "Tilt", "Defocus", "Oblique astigmatism", "Vertical astigmatism",
                     "Vertical coma", "Horizontal coma", "Vertical trefoil", "Oblique trefoil"]
    num_acts = len(sweep_params) // len(zernike_modes)

    i = 0  # Tip
    xs  = np.array([acts_sweep[i * num_acts + j][i] for j in range(num_acts)])
    ys  = np.mean(sweep_params[i * num_acts:(i + 1) * num_acts, :, i], axis=1)
    err = np.std( sweep_params[i * num_acts:(i + 1) * num_acts, :, i], axis=1)

    fig_width = 800 / 72
    fig_height = 800 / 72
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.errorbar(xs * slopes[i], ys - biases_saved[i], yerr=err, fmt='o', c='C0', label='Non-linear')
    ax.errorbar([-3, 3], [-3, 3], c='k', ls='-.', lw=1.5, alpha=0.7, zorder=-1, label='Ideal')
    ax.axhline(0, color='k', ls='--', lw=1.5, alpha=0.5)

    ax.set_title(rf"\textbf{{{zernike_modes[i]}}}", fontsize='medium')

    xmax = np.max(np.abs(xs * slopes[i])) * 1.4
    ticks = np.concatenate([np.arange(0, xmax + 0.5, 0.5), np.arange(0, -xmax - 0.5, -0.5)])
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlim([-xmax, xmax]); ax.set_ylim([-xmax, xmax])
    ax.set_xlabel('Input Zernike coefficients (rad)')
    ax.set_ylabel('Recovered Zernike coefficients (rad)')
    ax.legend(loc='lower right', frameon=True, fontsize='x-small')

    plt.subplots_adjust(bottom=0.28)
    plt.savefig('plots/presentation/lyot_sm_sweep_lab_tip_nonlinear_only.svg', transparent=True)
    plt.close()


def sweep_animation_gif():
    """Create a 10s animation of sweeping through the 9 Zernike modes once."""
    plt.style.use("configs/poster.mplstyle")
    
    # Load data for Lab: lyot small (standard sweep)
    data = pickle.load(open("data/flowfs_sweep_lyotsm_14.0mm_z_9modes_2025-11-30_12_05_58.pkl", "rb"))
    # data is a list of [f, actuators, surface, images] per step
    
    zernike_modes = [
        "Tip", "Tilt", "Defocus", "Oblique astigmatism", "Vertical astigmatism",
        "Vertical coma", "Horizontal coma", "Vertical trefoil", "Oblique trefoil"
    ]
    
    # Needs FLOWFS to process images
    flowfs = FLOWFS()
    config = json.load(open('./configs/flowfs_config_lyot_sm.json'))
    flowfs.projection(*config['projection'])
    flowfs.change_F_number(config['F_number_factor'] * flowfs.f_number)

    # Extract images for the sweep
    frames = []
    titles = []
    
    # Assume data is organized by mode
    num_modes = 9
    steps_per_mode = len(data) // num_modes
    
    # Process images and collect titles
    all_processed_ims = []
    for i in range(len(data)):
        f, actuators, surface, images = data[i]
        mode_idx = i // steps_per_mode
        if mode_idx >= num_modes:
            mode_idx = num_modes - 1
            
        # process_im(images[0], flowfs.binned_grid)
        im_processed = process_im(images[0], flowfs.binned_grid)
        all_processed_ims.append(im_processed)
        titles.append(f"Sweep: {zernike_modes[mode_idx]}")
    
    vmax = np.max([np.max(im) for im in all_processed_ims])
    frames = all_processed_ims
    
    num_frames = len(frames)
    duration_s = 10.0
    fps = num_frames / duration_s
    
    fig, ax = plt.subplots(figsize=(8, 8))
    # Use hcipy's imshow_field
    im = imshow_field(frames[0], cmap='inferno', vmin=0, vmax=vmax, ax=ax)
    plt.colorbar(im, label='Normalized Intensity', ax=ax, fraction=0.046, pad=0.04)
    title_text = ax.set_title(titles[0], fontsize='medium')
    ax.set_axis_off()
    
    def update(i):
        im.set_data(frames[i])
        title_text.set_text(titles[i])
        return [im, title_text]
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=True)
    
    # Save as GIF
    os.makedirs('plots/presentation', exist_ok=True)
    ani.save('plots/presentation/sweep_modes_animation.gif', writer='pillow', fps=fps)
    plt.close()


def presentation_single_tip_sweep():
    """Create a single-panel Tip sweep plot for presentation with crosstalk."""
    plt.style.use("configs/poster.mplstyle")
    # Use a square-ish aspect ratio for the single panel
    fig_width = 800 / 72
    fig_height = 800 / 72
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Data for Lab: lyot small
    sweep_params_linear = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params_linear.npy")
    sweep_params = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/sweep_params.npy")
    acts_sweep = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/acts.npy")

    zernike_modes = ["Tip", "Tilt", "Defocus", "Oblique astigmatism", "Vertical astigmatism", "Vertical coma", "Horizontal coma", "Vertical trefoil", "Oblique trefoil"]
    num_acts = len(sweep_params) // len(zernike_modes)

    # Calculate all slopes and biases to properly center the crosstalk plots
    slopes, biases = [], []
    for i in range(9):
        xs_i = np.array([acts_sweep[i * num_acts + j][i] for j in range(num_acts)])
        raw_i = sweep_params[i * num_acts:(i + 1) * num_acts, :, i]
        ys_i = np.mean(raw_i, axis=1)
        
        k = 3
        X = np.column_stack((xs_i[k:-k], np.ones(len(xs_i) - 2 * k)))
        slope_i = np.linalg.lstsq(X, ys_i[k:-k], rcond=None)[0][0]
        slopes.append(slope_i)
        biases.append(np.median(ys_i - xs_i * slope_i))

    i = 0  # Mode Tip
    xs = np.array([acts_sweep[i * num_acts + j][i] for j in range(num_acts)])
    ys = np.mean(sweep_params[i * num_acts:(i + 1) * num_acts, :, i], axis=1)
    err = np.std(sweep_params[i * num_acts:(i + 1) * num_acts, :, i], axis=1)
    
    ys_lin = np.mean(sweep_params_linear[i * num_acts:(i + 1) * num_acts, :, i], axis=1)
    err_lin = np.std(sweep_params_linear[i * num_acts:(i + 1) * num_acts, :, i], axis=1)

    # Plot Main
    ax.errorbar(xs * slopes[i], ys - biases[i], yerr=err, fmt='o', c='C0', label='Non-linear')
    ax.errorbar(xs * slopes[i], ys_lin, yerr=err_lin, fmt='^', c='C1', label='Linear')
    
    # Plot Crosstalk
    first_nl_ct = None
    first_l_ct = None
    for j in range(9):
        if j == i: continue
        y_nl_ct = np.mean(sweep_params[i * num_acts:(i + 1) * num_acts, :, j], axis=1) - biases[j]
        y_l_ct = np.mean(sweep_params_linear[i * num_acts:(i + 1) * num_acts, :, j], axis=1)
        h1 = ax.errorbar(xs * slopes[i], y_nl_ct, fmt='o', alpha=0.2, c='C2')
        h2 = ax.errorbar(xs * slopes[i], y_l_ct, fmt='^', alpha=0.2, c='C3')
        if first_nl_ct is None: first_nl_ct = h1
        if first_l_ct is None: first_l_ct = h2

    ax.errorbar([-3, 3], [-3, 3], c='k', ls='-.', lw=1.5, alpha=0.7, zorder=-1, label='Ideal')
    ax.axhline(0, color='k', ls='--', lw=1.5, alpha=0.5)

    # Labels and scale
    ax.set_xlabel('Input Zernike coefficients (rad)')
    ax.set_ylabel('Recovered Zernike coefficients (rad)')
    ax.set_title(rf"\textbf{{{zernike_modes[i]}}}", fontsize='medium')

    # Add bias and slope inset
    stats_text = (
        r"\begin{tabular}{@{}lr@{}}"
        rf"Bias: & {biases[i]:+.2f}\\"
        rf"Slope: & {slopes[i]:+.2f}"
        r"\end{tabular}"
    )
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
           ha='right', va='bottom', fontsize='x-small',
           bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7', alpha=0.9))

    xmax = np.max(np.abs(xs * slopes[i])) * 1.4
    ticks = np.concatenate([np.arange(0, xmax + 0.5, 0.5), np.arange(0, -xmax - 0.5, -0.5)])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-xmax, xmax])
    
    # Custom legend to include crosstalk
    handles, labels = ax.get_legend_handles_labels()
    # Find Ideal and order them
    ideal_idx = labels.index('Ideal')
    h_nl, h_l, h_ideal = handles[0], handles[1], handles[ideal_idx]
    
    ax.legend(
        [h_nl, first_nl_ct, h_l, first_l_ct, h_ideal],
        ['Non-linear', 'Crosstalk (non-linear)', 'Linear', 'Crosstalk (linear)', 'Ideal'],
        loc='lower center', ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.32),
        fontsize='x-small'
    )

    plt.subplots_adjust(bottom=0.28)
    plt.savefig('plots/presentation/lyot_sm_sweep_lab_tip_single.svg', transparent=True)
    plt.close()


def paper_reconstruction_plot():
    # Apply paper styling
    # plt.style.use("configs/paper.mplstyle")
    mpl.rcParams.update({
            "font.size": 10,
            "font.family": "sans-serif",
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{txfonts}",
            "legend.fontsize": "x-small",
            "legend.title_fontsize": "x-small",
            "xtick.labelsize": "x-small",
            "ytick.labelsize": "x-small",
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

    total_width = 523.5307 / 72  # set final figure width in inches
    data = pickle.load(open("data/flowfs_sweep_lyotsm_14.0mm_z_9modes_2025-11-30_12_05_58.pkl", "rb"))
    flowfs = FLOWFS()
    flowfs.load_config("configs/flowfs_lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab.json")
    flowfs.projection(1,1)
    im_WFS = data[5][3]
    

    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)
    wavefront = Wavefront(magellan_aperture, flowfs.wavelength_wfs)
    wavefront.total_power = 1

    res = phase_retrieval(flowfs, flowfs.start_theta, wavefront, im_WFS, plot=False, fit_downstream=True)

    flowfs.fitted_parameters[:] = True
    flowfs._set_aberration(res.x)
    pred = flowfs.forward(wavefront)
    print(im_WFS.min(), im_WFS.max(), pred.min(), pred.max())
    plot_difference(im_WFS, pred, ['Lab observation', 'Model', 'Residual'], figure_width=total_width)
    plt.savefig('plots/paper/lyot_sm_reconstruction.svg', transparent=True)
    plt.show()

# poster_sweep_plot()
    
def paper_lab_transition():
    mpl.rcParams.update({
            "font.size": 10,
            "font.family": "sans-serif",
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{txfonts}",
            "legend.fontsize": "x-small",
            "legend.title_fontsize": "x-small",
            "xtick.labelsize": "x-small",
            "ytick.labelsize": "x-small",
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

    total_width = 523.5307 / 72  # set final figure width in inches

    data = pickle.load(open('./data/flowfs_injection_modes_slow_control_14mm_2025-06-18_10_55_15.pkl', 'rb'))
    slopes = np.load("results/lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab/slopes.npy")
    
    flowfs = FLOWFS()
    flowfs.load_config("configs/flowfs_lyot_sm_14mm_z_9modes_2025-11-30_12_05_58_lab.json")

    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)

    fig, ax = plt.subplots(3,3, figsize = (total_width, total_width*.5), sharey=True, sharex=True)
    rms = []
    for j in range(len(data[:4])): 
        _, acts, ims, sci_ims = data[j]

        acts = np.array(acts) * slopes
        rms.append(flowfs.rms(acts[2,:9] - np.median(acts[-25:,:9], axis=0), magellan_aperture) * flowfs.wavelength_wfs / (2*np.pi) * 1e9)

        zernike_modes = [
            "Tip", "Tilt", "Defocus", "Oblique astigmatism", "Vertical astigmatism", "Vertical coma", "Horizontal coma", "Vertical trefoil", "Oblique trefoil"
        ]

        ax[7//3, 7%3].set_xlabel('Step')
        ax[3//3, 3%3].set_ylabel('Residual error (rad)')
        lim = np.max(np.abs(np.array(acts)[2:,] - np.median(np.array(acts)[-25:], axis=0))) *1.05
        for i in range(9):
            ax[i//3, i%3].axhline(0, ls='--', c='k', alpha=0.3, lw=0.75)
            if j == 0:
                ax[i//3, i%3].plot(np.array(acts)[2:,i] - np.median(np.array(acts)[-25:,i]))
            else:
                ax[i//3, i%3].plot(np.array(acts)[2:,i] - np.median(np.array(acts)[-25:,i]))
            # ax[i//3, i%3].set_title(zernike_modes[i])
            # b = np.median(np.array(acts)[50:,i])
            ax[i//3, i%3].set_ylim(-lim, lim)
            ax[i//3, i%3].set_xlim(0, 109)
            ax[i//3, i%3].legend([rf'$\textbf{{{zernike_modes[i]}}}$'], handlelength=0, handletextpad=0, frameon=False, fontsize='small')
            # new_bias.append(b)
            # plt.show()
    plt.subplots_adjust(wspace=0.05, hspace=.05, left=0.08, right=0.98, top=0.98, bottom=0.10)
    plt.savefig(f'plots/paper/lab_transition.svg')
    plt.show()
    print("rms", np.mean(rms), "nm")

#%%
if __name__ == "__main__":
    # sweep_animation_gif()
    # presentation_sweep_plot()
    # presentation_sweep_plot_nonlinear_only()
    # presentation_sweep_plot_both_clean()
    # presentation_single_tip_sweep()
    # presentation_single_tip_sweep_clean()
    # presentation_injection_retrieval_plots()
    # presentation_injection_retrieval_split()
    presentation_reconstruction_plot()
    # presentation_lab_transition()
    # sweep_plot()
    # paper_injection_retrieval_plots()
    # paper_reconstruction_plot()
    # injection_retrieval_plots()
    # poster_injection_retrieval_plots()
    # poster_sweep_plot()
    # paper_lab_transition()
    