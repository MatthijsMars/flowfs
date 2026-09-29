#!/usr/bin/env python3
import os

# Force single-threaded BLAS/OpenMP before importing numpy/scipy/hcipy
for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[k] = "1"

import argparse
import pickle
import time
import numpy as np
from tqdm.auto import tqdm

from hcipy import evaluate_supersampled, Wavefront, Field
from flowfs import FLOWFS, make_magaox_bump_mask, multi_phase_retrieval, phase_retrieval


def pin_to_cpu0():
    """Pin process to CPU core 0 on Linux (best effort)."""
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0})
            return True
        except Exception:
            return False
    return False


def make_flowfs(from_config: str, num_modes: int):
    flowfs = FLOWFS()
    flowfs.load_config(from_config)

    # Same defocus guard as calibration_from_sweep.py
    if flowfs.start_theta[-6] == 0:
        flowfs.start_theta[-6] = 18

    flowfs.create_upstream_aberration(num_modes)
    flowfs.fitted_parameters[:] = True
    flowfs._set_aberration(flowfs.start_theta)

    magaox_mask = make_magaox_bump_mask()
    magellan_aperture = evaluate_supersampled(magaox_mask, flowfs.pupil_grid, 6)
    wavefront = Wavefront(magellan_aperture, flowfs.wavelength_wfs)
    wavefront.total_power = 1

    return flowfs, wavefront


def load_images_for_sweep(
    sweep_file: str,
    binned_grid,
    num_modes: int,
    num_acts: int,
    num_frames: int,
    max_images: int | None,
):
    data = pickle.load(open(sweep_file, "rb"))

    im_refs = []
    for i in range(len(data)):
        _, _, _, images = data[i]
        im_refs.append(images)

    # Mirrors calibration_from_sweep.py layout
    a = np.array(im_refs).reshape(-1, num_modes, num_acts, 1024)
    a = a.swapaxes(0, 1).swapaxes(1, 2)  # -> (num_modes, num_acts, n_frames_available, 1024)

    ims_np = a[:num_modes, :, :num_frames].reshape(-1, 1024)
    if max_images is not None:
        ims_np = ims_np[:max_images]

    ims = [Field(im, binned_grid) for im in ims_np]
    return ims


def time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    return dt, out


def print_row(name: str, per_image_seconds):
    dts = np.asarray(per_image_seconds, dtype=float)
    mean_s = float(np.mean(dts))
    std_s = float(np.std(dts, ddof=1)) if dts.size > 1 else 0.0
    print(f"{name:42s} {mean_s:9.3f} ± {std_s:6.3f} s   ({1e3*mean_s:8.3f} ± {1e3*std_s:6.3f} ms/image)")


def main():
    parser = argparse.ArgumentParser(description="Single-core timing benchmark for FLOWFS phase retrieval.")
    parser.add_argument(
        "--sweep-file",
        type=str,
        default="data/flowfs_sweep_lyotsm_14.0mm_z_9modes_2025-11-30_12_05_58.pkl",
        help="Path to sweep pickle file (default: lyot_sm z-band lab sweep).",
    )
    parser.add_argument(
        "--from-config",
        type=str,
        default="./configs/flowfs_config_lyot_sm.json",
        help="Initial FLOWFS config json (default: lyot_sm base config).",
    )
    parser.add_argument("--num-modes", type=int, default=9)
    parser.add_argument("--num-acts", type=int, default=11)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--regularisation-strength", type=float, default=0.0)
    parser.add_argument("--probe-amp", type=float, default=0.3)
    parser.add_argument("--rcond", type=float, default=1e-2)
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images for faster test.")
    args = parser.parse_args()

    pinned = pin_to_cpu0()

    flowfs, wavefront = make_flowfs(args.from_config, args.num_modes)
    ims = load_images_for_sweep(
        args.sweep_file,
        flowfs.binned_grid,
        args.num_modes,
        args.num_acts,
        args.num_frames,
        args.max_images,
    )

    base_theta = flowfs.start_theta.copy()
    n_images = len(ims)

    print(f"Pinned to CPU0: {pinned}")
    print(f"Images used:    {n_images}")
    print("Std is computed across images (single run, no repeats).")
    print("-" * 80)

    # 1) Non-linear: all parameters (upstream + downstream)
    dt_all_per_image = []
    for im in tqdm(ims):
        dt, _ = time_call(
            lambda im=im: phase_retrieval(
                flowfs,
                base_theta.copy(),
                wavefront,
                im,
                plot=False,
                fix_downstream_tip_tilt=True,
                fit_downstream=True,
                regularisation_strength=args.regularisation_strength,
                # num_workers=1,
            )
        )
        dt_all_per_image.append(dt)

    # 2) Non-linear: upstream only
    dt_up_nl_per_image = []
    for im in tqdm(ims):
        dt, _ = time_call(
            lambda im=im: phase_retrieval(
                flowfs,
                base_theta.copy(),
                wavefront,
                im,
                plot=False,
                fix_downstream_tip_tilt=False,
                fit_downstream=False,
                regularisation_strength=args.regularisation_strength,
                # num_workers=1,
            )
        )
        dt_up_nl_per_image.append(dt)

    # 3) Linear: upstream only
    dt_lin_setup, (_, linear_solver, _) = time_call(
        lambda: flowfs.get_reconstruction_matrix(
            wavefront,
            probe_amp=args.probe_amp,
            rcond=args.rcond,
        )
    )
    dt_lin_apply_per_image = []
    for im in tqdm(ims):
        dt, _ = time_call(lambda im=im: linear_solver(im))
        dt_lin_apply_per_image.append(dt)

    # Amortize one-time setup across images for a fair per-image total
    dt_lin_total_per_image = np.asarray(dt_lin_apply_per_image) + (dt_lin_setup / max(1, n_images))

    print_row("1) Non-linear (upstream + downstream)", dt_all_per_image)
    print_row("2) Non-linear (upstream only)", dt_up_nl_per_image)
    print_row("3) Linear (upstream only, total)", dt_lin_total_per_image)
    print_row("   └─ linear apply only", dt_lin_apply_per_image)
    print(f"{'   └─ linear setup (one-time)':42s} {dt_lin_setup:9.3f} s")


if __name__ == "__main__":
    main()